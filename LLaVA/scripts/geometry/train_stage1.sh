#!/bin/bash

set -euo pipefail


# ============================================================
# Stage 1: Primitive Geometry Projector Training
#
# CLIP vision encoder : frozen
# MLP projector       : trainable
# LLM backbone        : frozen
#
# train-only
# 1 epoch
# validation 없음
#
# Stage 1의 checkpoint-*는 LLaVA adapter 전용 저장 구조상
# 완전한 optimizer resume checkpoint가 아니다.
# 따라서 중간 checkpoint 저장은 사용하지 않고,
# 정상 종료 시 최종 mm_projector.bin만 저장한다.
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
# 데이터
# ------------------------------------------------------------

TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-./geometry_data/stage1_geometry_grounding/train.json}"

IMAGE_FOLDER="${IMAGE_FOLDER:-./geometry_data/stage1_geometry_grounding/images}"


# ------------------------------------------------------------
# 출력
# ------------------------------------------------------------

OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/geometry_stage1}"


# ------------------------------------------------------------
# GPU / DeepSpeed
# ------------------------------------------------------------

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MASTER_PORT="${MASTER_PORT:-29501}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-./scripts/zero2.json}"


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
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"


# ------------------------------------------------------------
# 학습 하이퍼파라미터
#
# 4,002 samples / global batch 32
# ≈ 125 optimizer steps
#
# 기존 1e-5는 변화가 거의 없을 가능성이 높아
# projector 기본 LR을 1e-4로 둔다.
# LR 비교 실험은 반드시 다른 OUTPUT_DIR에서 수행한다.
# ------------------------------------------------------------

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"

LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MM_PROJECTOR_LR="${MM_PROJECTOR_LR:-1e-4}"

WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# -1이면 전체 epoch을 수행한다.
# smoke test에서는 양수로 덮어쓴다.
MAX_STEPS="${MAX_STEPS:--1}"

MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"


# ------------------------------------------------------------
# 저장
#
# 최종 mm_projector.bin만 저장한다.
# ------------------------------------------------------------

SAVE_STRATEGY="${SAVE_STRATEGY:-no}"


# ------------------------------------------------------------
# 메모리 / DataLoader
# ------------------------------------------------------------

GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"


# ------------------------------------------------------------
# 필수 입력 확인
# ------------------------------------------------------------

for FILE_PATH in \
    "${TRAIN_DATA_PATH}" \
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
# output directory 안전 검사
#
# 업스트림 LLaVA train()은 output_dir에 checkpoint-*가 있으면
# 자동으로 resume를 시도한다.
#
# Stage 1 adapter checkpoint는 완전한 resume checkpoint가
# 아니므로, 비어 있지 않은 output directory는 허용하지 않는다.
# ------------------------------------------------------------

if [[ -d "${OUTPUT_DIR}" ]] \
    && [[ -n "$(
        find "${OUTPUT_DIR}" \
            -mindepth 1 \
            -maxdepth 1 \
            -print \
            -quit
    )" ]]; then

    echo "Stage 1 output directory is not empty:" >&2
    echo "${OUTPUT_DIR}" >&2
    echo "Use a new OUTPUT_DIR or remove the old output explicitly." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"


# ------------------------------------------------------------
# 설정 출력
# ------------------------------------------------------------

echo "Starting Stage 1 projector training."
echo "Model: ${MODEL_PATH}"
echo "Vision tower: ${VISION_TOWER}"
echo "Training data: ${TRAIN_DATA_PATH}"
echo "Image folder: ${IMAGE_FOLDER}"
echo "Output directory: ${OUTPUT_DIR}"
echo "DeepSpeed config: ${DEEPSPEED_CONFIG}"
echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Number of GPUs: ${NUM_GPUS}"
echo "Per-device train batch size: ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "Gradient accumulation steps: ${GRADIENT_ACCUMULATION_STEPS}"
echo "Effective global batch size: ${GLOBAL_BATCH_SIZE}"
echo "Training epochs: ${NUM_TRAIN_EPOCHS}"
echo "Maximum steps: ${MAX_STEPS}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Projector learning rate: ${MM_PROJECTOR_LR}"
echo "Evaluation: disabled"
echo "Save strategy: ${SAVE_STRATEGY}"
echo "Logging steps: ${LOGGING_STEPS}"
echo "BF16: ${BF16}"
echo "FP16: ${FP16}"
echo "TF32: ${TF32}"
echo "Attention implementation: ${ATTENTION_IMPLEMENTATION}"
echo "Gradient checkpointing: ${GRADIENT_CHECKPOINTING}"


# ------------------------------------------------------------
# Stage 1 학습
# ------------------------------------------------------------

deepspeed \
    --master_port "${MASTER_PORT}" \
    llava/train/train_stage1.py \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --model_name_or_path "${MODEL_PATH}" \
    --version "v1" \
    --data_path "${TRAIN_DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --vision_tower "${VISION_TOWER}" \
    --mm_projector_type "mlp2x_gelu" \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature "patch" \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio "pad" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --evaluation_strategy "no" \
    --save_strategy "${SAVE_STRATEGY}" \
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


# ------------------------------------------------------------
# 최종 projector 확인
# ------------------------------------------------------------

FINAL_PROJECTOR="${OUTPUT_DIR}/mm_projector.bin"

if [[ ! -f "${FINAL_PROJECTOR}" ]]; then
    echo "Stage 1 training finished, but the projector was not saved." >&2
    echo "Expected path: ${FINAL_PROJECTOR}" >&2
    exit 1
fi


echo "Stage 1 projector training completed successfully."
echo "Final projector: ${FINAL_PROJECTOR}"