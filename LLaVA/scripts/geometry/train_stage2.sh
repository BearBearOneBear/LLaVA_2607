#!/bin/bash

set -euo pipefail


# ============================================================
# Stage 2: Geometry Relation and Property Training
#
# CLIP vision encoder : frozen
# MLP projector       : trainable
# LLM backbone        : trainable
# ============================================================


is_true() {
    case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
        true|1|yes|y|on)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}


# ------------------------------------------------------------
# 모델
# ------------------------------------------------------------

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
VISION_TOWER="${VISION_TOWER:-openai/clip-vit-large-patch14-336}"


# ------------------------------------------------------------
# Stage 1 projector
# ------------------------------------------------------------

STAGE1_PROJECTOR_PATH="${STAGE1_PROJECTOR_PATH:-./checkpoints/geometry_stage1/mm_projector.bin}"


# ------------------------------------------------------------
# 데이터
# ------------------------------------------------------------

TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-./geometry_data/stage2_geometry_grounding/train.json}"

EVAL_DATA_PATH="${EVAL_DATA_PATH:-./geometry_data/stage2_geometry_grounding/validation.json}"

IMAGE_FOLDER="${IMAGE_FOLDER:-./geometry_data/stage2_geometry_grounding/images}"


# ------------------------------------------------------------
# 출력
# ------------------------------------------------------------

OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/geometry_stage2}"


# ------------------------------------------------------------
# GPU / DeepSpeed
# ------------------------------------------------------------

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MASTER_PORT="${MASTER_PORT:-29502}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-./scripts/zero3_offload.json}"


# ------------------------------------------------------------
# 정밀도
#
# RTX 5090 / A100 / RTX 4090 / H100 등:
#   BF16=True
#   FP16=False
#   TF32=True
#
# V100 / T4 등:
#   BF16=False
#   FP16=True
#   TF32=False
#   ATTENTION_IMPLEMENTATION=sdpa 권장
# ------------------------------------------------------------

BF16="${BF16:-True}"
FP16="${FP16:-False}"
TF32="${TF32:-True}"

if is_true "${BF16}" && is_true "${FP16}"; then
    echo "BF16 and FP16 cannot both be enabled." >&2
    exit 1
fi


# ------------------------------------------------------------
# Attention backend
#
# auto:
#   FlashAttention-2 사용 가능 -> flash_attention_2
#   FlashAttention-2 사용 불가 -> sdpa
#   SDPA도 사용 불가          -> Transformers 기본 구현
#
# 가능한 값:
#   auto
#   flash_attention_2
#   sdpa
#   eager
#   none
#   default
# ------------------------------------------------------------

export ATTENTION_IMPLEMENTATION="${ATTENTION_IMPLEMENTATION:-auto}"


# ------------------------------------------------------------
# Batch
#
# effective global batch:
# GPU 수
# × per-device train batch
# × gradient accumulation
# ------------------------------------------------------------

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"


# ------------------------------------------------------------
# 학습 하이퍼파라미터
# ------------------------------------------------------------

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"

LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MM_PROJECTOR_LR="${MM_PROJECTOR_LR:-3e-5}"

WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# -1이면 전체 epoch을 수행한다.
# smoke test에서는 양수로 덮어쓴다.
MAX_STEPS="${MAX_STEPS:--1}"

MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"


# ------------------------------------------------------------
# Validation / checkpoint / logging
# ------------------------------------------------------------

EVALUATION_STRATEGY="${EVALUATION_STRATEGY:-epoch}"
SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"

# strategy=steps일 때만 실질적으로 사용된다.
EVAL_STEPS="${EVAL_STEPS:-20}"
SAVE_STEPS="${SAVE_STEPS:-20}"

SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"

LOAD_BEST_MODEL_AT_END="${LOAD_BEST_MODEL_AT_END:-True}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-False}"

# False:
#   output_dir에 checkpoint-*가 있으면 중단한다.
#
# True:
#   업스트림 LLaVA의 자동 resume를 명시적으로 허용한다.

# ZeRO-3 full fine-tuning checkpoint 두 개와 최종 모델을
# 고려한 보수적인 기본값이다.
#
# 0으로 설정하면 디스크 검사를 비활성화한다.
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-220}"


# ------------------------------------------------------------
# 메모리 / DataLoader
# ------------------------------------------------------------

GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"


# ------------------------------------------------------------
# 필수 입력 확인
# ------------------------------------------------------------

for FILE_PATH in \
    "${STAGE1_PROJECTOR_PATH}" \
    "${TRAIN_DATA_PATH}" \
    "${EVAL_DATA_PATH}" \
    "${DEEPSPEED_CONFIG}"
do
    if [[ ! -f "${FILE_PATH}" ]]; then
        echo "Required file was not found: ${FILE_PATH}" >&2
        exit 1
    fi
done

if [[ ! -d "${IMAGE_FOLDER}" ]]; then
    echo "Image directory was not found: ${IMAGE_FOLDER}" >&2
    exit 1
fi

if ! command -v deepspeed >/dev/null 2>&1; then
    echo "The deepspeed command was not found." >&2
    exit 1
fi


# ------------------------------------------------------------
# GPU 수 확인
# ------------------------------------------------------------

if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
    NUM_GPUS=0
else
    IFS=',' read -ra GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
    NUM_GPUS="${#GPU_ARRAY[@]}"
fi

if [[ "${NUM_GPUS}" -eq 0 ]]; then
    echo "No GPU is visible. Set CUDA_VISIBLE_DEVICES." >&2
    exit 1
fi

GLOBAL_BATCH_SIZE=$((
    NUM_GPUS
    * PER_DEVICE_TRAIN_BATCH_SIZE
    * GRADIENT_ACCUMULATION_STEPS
))


# ------------------------------------------------------------
# Trainer 설정 호환성 확인
# ------------------------------------------------------------

if is_true "${LOAD_BEST_MODEL_AT_END}"; then
    if [[ "${EVALUATION_STRATEGY}" == "no" ]] \
        || [[ "${SAVE_STRATEGY}" == "no" ]]; then

        echo "load_best_model_at_end requires evaluation and saving." >&2
        exit 1
    fi

    if [[ "${EVALUATION_STRATEGY}" != "${SAVE_STRATEGY}" ]]; then
        echo "load_best_model_at_end requires matching strategies." >&2
        echo "evaluation_strategy=${EVALUATION_STRATEGY}" >&2
        echo "save_strategy=${SAVE_STRATEGY}" >&2
        exit 1
    fi

    if [[ "${SAVE_STRATEGY}" == "steps" ]]; then
        if (( EVAL_STEPS <= 0 || SAVE_STEPS <= 0 )); then
            echo "EVAL_STEPS and SAVE_STEPS must be positive." >&2
            exit 1
        fi

        if (( SAVE_STEPS % EVAL_STEPS != 0 )); then
            echo "SAVE_STEPS must be a multiple of EVAL_STEPS." >&2
            exit 1
        fi
    fi
fi


# ------------------------------------------------------------
# 묵시적 auto-resume 방지
#
# 업스트림 LLaVA train()은 output_dir에 checkpoint-*가
# 하나라도 있으면 자동으로 resume_from_checkpoint=True를
# 사용한다.
# ------------------------------------------------------------

EXISTING_CHECKPOINT=""

if [[ -d "${OUTPUT_DIR}" ]]; then
    EXISTING_CHECKPOINT="$(
        find "${OUTPUT_DIR}" \
            -maxdepth 1 \
            -type d \
            -name 'checkpoint-*' \
            -print \
            -quit \
            2>/dev/null
    )"
fi

if [[ -n "${EXISTING_CHECKPOINT}" ]]; then
    if ! is_true "${RESUME_FROM_CHECKPOINT}"; then
        echo "Existing Stage 2 checkpoints were found:" >&2
        echo "${EXISTING_CHECKPOINT}" >&2
        echo "The upstream trainer would resume automatically." >&2
        echo "Use a new OUTPUT_DIR or set" >&2
        echo "RESUME_FROM_CHECKPOINT=True." >&2
        exit 1
    fi

    echo "Resuming from an existing Stage 2 checkpoint."
else
    if is_true "${RESUME_FROM_CHECKPOINT}"; then
        echo "RESUME_FROM_CHECKPOINT=True was requested," >&2
        echo "but no checkpoint-* directory was found." >&2
        exit 1
    fi

    if [[ -d "${OUTPUT_DIR}" ]] \
        && [[ -n "$(
            find "${OUTPUT_DIR}" \
                -mindepth 1 \
                -maxdepth 1 \
                -print \
                -quit
        )" ]]; then

        echo "Stage 2 output directory is not empty:" >&2
        echo "${OUTPUT_DIR}" >&2
        echo "Use a new OUTPUT_DIR or remove the old output" >&2
        echo "explicitly." >&2
        exit 1
    fi
fi


# ------------------------------------------------------------
# 디스크 여유 공간 확인
# ------------------------------------------------------------

if ! [[ "${MIN_FREE_DISK_GB}" =~ ^[0-9]+$ ]]; then
    echo "MIN_FREE_DISK_GB must be a non-negative integer." >&2
    exit 1
fi

OUTPUT_PARENT="$(dirname "${OUTPUT_DIR}")"
mkdir -p "${OUTPUT_PARENT}"

AVAILABLE_KB="$(
    df -Pk "${OUTPUT_PARENT}" \
        | awk 'NR == 2 {print $4}'
)"

if [[ -z "${AVAILABLE_KB}" ]]; then
    echo "Could not determine free disk space:" >&2
    echo "${OUTPUT_PARENT}" >&2
    exit 1
fi

AVAILABLE_GB=$((AVAILABLE_KB / 1024 / 1024))

if (( MIN_FREE_DISK_GB > 0 )) \
    && (( AVAILABLE_GB < MIN_FREE_DISK_GB )); then

    echo "Insufficient free disk space for Stage 2." >&2
    echo "Available: ${AVAILABLE_GB} GB" >&2
    echo "Required: ${MIN_FREE_DISK_GB} GB" >&2
    exit 1
fi


mkdir -p "${OUTPUT_DIR}"


# ------------------------------------------------------------
# 설정 출력
# ------------------------------------------------------------

echo "Starting Stage 2 LLM and projector training."
echo "Model: ${MODEL_PATH}"
echo "Vision tower: ${VISION_TOWER}"
echo "Stage 1 projector: ${STAGE1_PROJECTOR_PATH}"
echo "Training data: ${TRAIN_DATA_PATH}"
echo "Validation data: ${EVAL_DATA_PATH}"
echo "Image folder: ${IMAGE_FOLDER}"
echo "Output directory: ${OUTPUT_DIR}"
echo "DeepSpeed config: ${DEEPSPEED_CONFIG}"
echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Number of GPUs: ${NUM_GPUS}"
echo "Per-device train batch size: ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "Per-device eval batch size: ${PER_DEVICE_EVAL_BATCH_SIZE}"
echo "Gradient accumulation steps: ${GRADIENT_ACCUMULATION_STEPS}"
echo "Effective global batch size: ${GLOBAL_BATCH_SIZE}"
echo "LLM learning rate: ${LEARNING_RATE}"
echo "Projector learning rate: ${MM_PROJECTOR_LR}"
echo "Number of epochs: ${NUM_TRAIN_EPOCHS}"
echo "Maximum steps: ${MAX_STEPS}"
echo "Evaluation strategy: ${EVALUATION_STRATEGY}"
echo "Save strategy: ${SAVE_STRATEGY}"
echo "Save total limit: ${SAVE_TOTAL_LIMIT}"
echo "Load best model at end: ${LOAD_BEST_MODEL_AT_END}"
echo "Resume from checkpoint: ${RESUME_FROM_CHECKPOINT}"
echo "Logging steps: ${LOGGING_STEPS}"
echo "BF16: ${BF16}"
echo "FP16: ${FP16}"
echo "TF32: ${TF32}"
echo "Attention implementation: ${ATTENTION_IMPLEMENTATION}"
echo "Gradient checkpointing: ${GRADIENT_CHECKPOINTING}"
echo "Available disk space: ${AVAILABLE_GB} GB"
echo "Required free disk space: ${MIN_FREE_DISK_GB} GB"


# ------------------------------------------------------------
# Stage 2 학습
# ------------------------------------------------------------

deepspeed \
    --master_port "${MASTER_PORT}" \
    llava/train/train_stage2.py \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --model_name_or_path "${MODEL_PATH}" \
    --version "v1" \
    --data_path "${TRAIN_DATA_PATH}" \
    --eval_data_path "${EVAL_DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --vision_tower "${VISION_TOWER}" \
    --pretrain_mm_mlp_adapter "${STAGE1_PROJECTOR_PATH}" \
    --mm_projector_type "mlp2x_gelu" \
    --freeze_backbone False \
    --tune_mm_mlp_adapter False \
    --freeze_mm_mlp_adapter False \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature "patch" \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio "pad" \
    --group_by_modality_length True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --evaluation_strategy "${EVALUATION_STRATEGY}" \
    --eval_steps "${EVAL_STEPS}" \
    --save_strategy "${SAVE_STRATEGY}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --load_best_model_at_end "${LOAD_BEST_MODEL_AT_END}" \
    --metric_for_best_model "eval_loss" \
    --greater_is_better False \
    --learning_rate "${LEARNING_RATE}" \
    --mm_projector_lr "${MM_PROJECTOR_LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --lr_scheduler_type "cosine" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --logging_steps "${LOGGING_STEPS}" \
    --logging_first_step True \
    --logging_nan_inf_filter False \
    --bf16 "${BF16}" \
    --fp16 "${FP16}" \
    --tf32 "${TF32}" \
    --bits 16 \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --lazy_preprocess True \
    --report_to "none"


echo "Stage 2 LLM and projector training completed."
echo "Best Stage 2 model directory: ${OUTPUT_DIR}"