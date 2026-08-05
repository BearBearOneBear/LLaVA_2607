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
# ============================================================


# ------------------------------------------------------------
# 모델
# ------------------------------------------------------------

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
VISION_TOWER="${VISION_TOWER:-openai/clip-vit-large-patch14-336}"


# ------------------------------------------------------------
# 데이터 경로
#
# convert_parquet.sh의 실제 출력 경로와 일치시킨다.
# ------------------------------------------------------------

TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-./geometry_data/stage1_geometry_grounding/train.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-./geometry_data/stage1_geometry_grounding/images}"


# ------------------------------------------------------------
# 출력 경로
# ------------------------------------------------------------

OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/geometry_stage1}"


# ------------------------------------------------------------
# GPU / DeepSpeed
# ------------------------------------------------------------

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MASTER_PORT="${MASTER_PORT:-29501}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-./scripts/zero2.json}"


# ------------------------------------------------------------
# 정밀도
# ------------------------------------------------------------

BF16="${BF16:-True}"
FP16="${FP16:-False}"
TF32="${TF32:-True}"


# ------------------------------------------------------------
# Batch
#
# effective global batch:
# GPU 수 × per-device batch × gradient accumulation
# ------------------------------------------------------------

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"


# ------------------------------------------------------------
# 학습 하이퍼파라미터
# ------------------------------------------------------------

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"

LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MM_PROJECTOR_LR="${MM_PROJECTOR_LR:-1e-5}"

WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# -1이면 전체 epoch을 수행한다.
# smoke test에서는 양수 값으로 덮어쓴다.
MAX_STEPS="${MAX_STEPS:--1}"

MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"

LOGGING_STEPS="${LOGGING_STEPS:-1}"


# ------------------------------------------------------------
# 메모리 설정
# ------------------------------------------------------------

GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"


# ------------------------------------------------------------
# 필수 입력 확인
# ------------------------------------------------------------

if [[ ! -f "${TRAIN_DATA_PATH}" ]]; then
    echo "Stage 1 train JSON was not found:" >&2
    echo "${TRAIN_DATA_PATH}" >&2
    exit 1
fi

if [[ ! -d "${IMAGE_FOLDER}" ]]; then
    echo "Stage 1 image directory was not found:" >&2
    echo "${IMAGE_FOLDER}" >&2
    exit 1
fi

if [[ ! -f "${DEEPSPEED_CONFIG}" ]]; then
    echo "DeepSpeed config was not found:" >&2
    echo "${DEEPSPEED_CONFIG}" >&2
    exit 1
fi

if ! command -v deepspeed >/dev/null 2>&1; then
    echo "The deepspeed command was not found." >&2
    exit 1
fi


# ------------------------------------------------------------
# 설정 출력
# ------------------------------------------------------------

IFS=',' read -ra GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_ARRAY[@]}"

GLOBAL_BATCH_SIZE=$((
    NUM_GPUS
    * PER_DEVICE_TRAIN_BATCH_SIZE
    * GRADIENT_ACCUMULATION_STEPS
))

echo "Starting Stage 1 projector training."
echo "Model: ${MODEL_PATH}"
echo "Training data: ${TRAIN_DATA_PATH}"
echo "Image folder: ${IMAGE_FOLDER}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Training epochs: ${NUM_TRAIN_EPOCHS}"
echo "Maximum steps: ${MAX_STEPS}"
echo "Evaluation: disabled"
echo "DeepSpeed config: ${DEEPSPEED_CONFIG}"
echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Number of GPUs: ${NUM_GPUS}"
echo "Per-device train batch size: ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "Gradient accumulation steps: ${GRADIENT_ACCUMULATION_STEPS}"
echo "Effective global batch size: ${GLOBAL_BATCH_SIZE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Projector learning rate: ${MM_PROJECTOR_LR}"
echo "BF16: ${BF16}"
echo "FP16: ${FP16}"
echo "TF32: ${TF32}"


mkdir -p "${OUTPUT_DIR}"


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
    --save_strategy "no" \
    --learning_rate "${LEARNING_RATE}" \
    --mm_projector_lr "${MM_PROJECTOR_LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --lr_scheduler_type "cosine" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --logging_steps "${LOGGING_STEPS}" \
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
# 최종 projector 저장 확인
# ------------------------------------------------------------

FINAL_PROJECTOR="${OUTPUT_DIR}/mm_projector.bin"

if [[ ! -f "${FINAL_PROJECTOR}" ]]; then
    echo "Stage 1 training finished, but the projector was not saved." >&2
    echo "Expected path: ${FINAL_PROJECTOR}" >&2
    exit 1
fi

echo "Stage 1 projector training completed."
echo "Final projector: ${FINAL_PROJECTOR}"