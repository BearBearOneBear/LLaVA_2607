#!/bin/bash

set -euo pipefail


# ============================================================
# Stage 1: Primitive Geometry Projector Training
# CLIP vision encoder   frozen
# MLP                   trainable
# LLM backbone          frozen
# ============================================================


# ------------------------------------------------------------
# model
# ------------------------------------------------------------

# LLaVA-1.5 7B checkpoint (G-LLaVA와 동일)
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"

# LLaVA-1.5 CLIP vision encoder
VISION_TOWER="${VISION_TOWER:-openai/clip-vit-large-patch14-336}"


# ------------------------------------------------------------
# data path
# ------------------------------------------------------------

# train
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-./geometry_data/stage1/train.json}"

# validation
EVAL_DATA_PATH="${EVAL_DATA_PATH:-./geometry_data/stage1/validation.json}"

# image
IMAGE_FOLDER="${IMAGE_FOLDER:-./geometry_data/stage1/images}"


# ------------------------------------------------------------
# output path
# ------------------------------------------------------------

# mlp checkpoint, trainer state
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/geometry_stage1}"


# ------------------------------------------------------------
# GPU 개수
#
# GPU 한 장:
#   CUDA_VISIBLE_DEVICES=0
#
# GPU 두 장:
#   CUDA_VISIBLE_DEVICES=0,1
#
# deepspeed는 CUDA_VISIBLE_DEVICES에 지정된 GPU를 모두 사용
# ------------------------------------------------------------

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MASTER_PORT="${MASTER_PORT:-29501}"

DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-./scripts/zero2.json}"


# ------------------------------------------------------------
# 정밀도 설정
#
# A100, A6000, RTX 3090/4090, L40, H100 등:
#   BF16=True
#   FP16=False
#   TF32=True
#
# V100, T4 등:
#   BF16=False
#   FP16=True
#   TF32=False
#
# ------------------------------------------------------------

BF16="${BF16:-True}"
FP16="${FP16:-False}"
TF32="${TF32:-True}"


# ------------------------------------------------------------
# Batch 설정
#
# 메모리에 따라 PER_DEVICE_TRAIN_BATCH_SIZE 설정
#
# global batch:
#   GPU 수
#   × PER_DEVICE_TRAIN_BATCH_SIZE
#   × GRADIENT_ACCUMULATION_STEPS
# ------------------------------------------------------------

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"


# ------------------------------------------------------------
# 학습 하이퍼 파라미터
# ------------------------------------------------------------

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-5}"

# checkpoint에서 시작 -> llava pretraining (1e-3)보다 작은 값
LEARNING_RATE="${LEARNING_RATE:-1e-5}"

MM_PROJECTOR_LR="${MM_PROJECTOR_LR:-1e-5}"

WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# 학습 step 수, -1은 전체
MAX_STEPS="${MAX_STEPS:--1}"


# ------------------------------------------------------------
# Validation 및 checkpoint 설정
#
# 데이터 2250, global batcch 32 -> 1 epoch : 71 step
#
# eval_Steps=10 -> epoch 당 7번 validation
# ------------------------------------------------------------

EVAL_STEPS="${EVAL_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-10}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"


# ------------------------------------------------------------
# 메모리 설정
#
# True:
#   GPU 메모리 부족
#
# False:
#   GPU 메모리 충분
# ------------------------------------------------------------

GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"

# CPU worker 수
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"


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
echo "Validation data: ${EVAL_DATA_PATH}"
echo "Image folder: ${IMAGE_FOLDER}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Number of GPUs: ${NUM_GPUS}"
echo "Per-device train batch size: ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "Gradient accumulation steps: ${GRADIENT_ACCUMULATION_STEPS}"
echo "Effective global batch size: ${GLOBAL_BATCH_SIZE}"
echo "BF16: ${BF16}"
echo "FP16: ${FP16}"
echo "TF32: ${TF32}"


# output 폴더
mkdir -p "${OUTPUT_DIR}"


# ------------------------------------------------------------
# Stage 1 학습 실행
# ------------------------------------------------------------

deepspeed \
    --master_port "${MASTER_PORT}" \
    llava/train/train_stage1.py \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --model_name_or_path "${MODEL_PATH}" \
    --version "v1" \
    --data_path "${TRAIN_DATA_PATH}" \
    --eval_data_path "${EVAL_DATA_PATH}" \
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
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --evaluation_strategy "steps" \
    --eval_steps "${EVAL_STEPS}" \
    --save_strategy "steps" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
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
    --model_max_length 512 \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --lazy_preprocess True \
    --report_to "none"


echo "Stage 1 projector training completed."