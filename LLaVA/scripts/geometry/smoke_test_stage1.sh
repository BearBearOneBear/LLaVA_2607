#!/bin/bash

set -euo pipefail


# ============================================================
# Stage 1 Smoke Test
# - Model Load
# - Data read
# - Loss
# - checkpoint
# ============================================================


# ------------------------------------------------------------
# data path
# ------------------------------------------------------------

TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-./geometry_data/stage1/train.json}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-./geometry_data/stage1/validation.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-./geometry_data/stage1/images}"


# ------------------------------------------------------------
# output path
# ------------------------------------------------------------

SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-./checkpoints/geometry_stage1_smoke}"

rm -rf "${SMOKE_OUTPUT_DIR}"
mkdir -p "${SMOKE_OUTPUT_DIR}"



echo "Starting Stage 1 smoke test."
echo "Training data: ${TRAIN_DATA_PATH}"
echo "Validation data: ${EVAL_DATA_PATH}"
echo "Image folder: ${IMAGE_FOLDER}"
echo "Output directory: ${SMOKE_OUTPUT_DIR}"


# ------------------------------------------------------------
# 기존 Stage 1 실행 스크립트를 재사용
# ------------------------------------------------------------

TRAIN_DATA_PATH="${TRAIN_DATA_PATH}" \
EVAL_DATA_PATH="${EVAL_DATA_PATH}" \
IMAGE_FOLDER="${IMAGE_FOLDER}" \
OUTPUT_DIR="${SMOKE_OUTPUT_DIR}" \
MAX_STEPS=2 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
PER_DEVICE_EVAL_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=1 \
EVAL_STEPS=1 \
SAVE_STEPS=1 \
SAVE_TOTAL_LIMIT=2 \
LOGGING_STEPS=1 \
DATALOADER_NUM_WORKERS=0 \
bash scripts/geometry/train_stage1.sh


echo "Stage 1 smoke test completed."
echo "Check trainer state: ${SMOKE_OUTPUT_DIR}/trainer_state.json"
echo "Check final projector: ${SMOKE_OUTPUT_DIR}/mm_projector.bin"
echo "Check intermediate checkpoints: ${SMOKE_OUTPUT_DIR}/checkpoint-*"