#!/bin/bash

set -euo pipefail


# ============================================================
# Stage 2 Smoke Test
#   - Model load
#   - MLP projector load
#   - Data read
#   - Backward
#   - Loss
#   - Checkpoint
# ============================================================


# ------------------------------------------------------------
# Stage 1 projector
# ------------------------------------------------------------

STAGE1_PROJECTOR_PATH="${STAGE1_PROJECTOR_PATH:-./checkpoints/geometry_stage1/mm_projector.bin}"

if [[ ! -f "${STAGE1_PROJECTOR_PATH}" ]]; then
    echo "Stage 1 projector was not found:" >&2
    echo "${STAGE1_PROJECTOR_PATH}" >&2
    exit 1
fi


# ------------------------------------------------------------
# Data paths
# ------------------------------------------------------------

TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-./geometry_data/stage2_geometry_grounding/train.json}"

EVAL_DATA_PATH="${EVAL_DATA_PATH:-./geometry_data/stage2_geometry_grounding/validation.json}"

IMAGE_FOLDER="${IMAGE_FOLDER:-./geometry_data/stage2_geometry_grounding/images}"


# ------------------------------------------------------------
# Output path
# ------------------------------------------------------------

SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-./checkpoints/geometry_stage2_smoke}"

# 기본 smoke-test 경로 또는 명시적으로 지정한 경로만 삭제한다.
if [[ -z "${SMOKE_OUTPUT_DIR}" ]] || [[ "${SMOKE_OUTPUT_DIR}" == "/" ]]; then
    echo "Unsafe SMOKE_OUTPUT_DIR: ${SMOKE_OUTPUT_DIR}" >&2
    exit 1
fi

rm -rf "${SMOKE_OUTPUT_DIR}"
mkdir -p "${SMOKE_OUTPUT_DIR}"


echo "Starting Stage 2 smoke test."
echo "Stage 1 projector: ${STAGE1_PROJECTOR_PATH}"
echo "Training data: ${TRAIN_DATA_PATH}"
echo "Validation data: ${EVAL_DATA_PATH}"
echo "Image folder: ${IMAGE_FOLDER}"
echo "Output directory: ${SMOKE_OUTPUT_DIR}"


# ------------------------------------------------------------
# Reuse the Stage 2 training script
# ------------------------------------------------------------

STAGE1_PROJECTOR_PATH="${STAGE1_PROJECTOR_PATH}" \
TRAIN_DATA_PATH="${TRAIN_DATA_PATH}" \
EVAL_DATA_PATH="${EVAL_DATA_PATH}" \
IMAGE_FOLDER="${IMAGE_FOLDER}" \
OUTPUT_DIR="${SMOKE_OUTPUT_DIR}" \
NUM_TRAIN_EPOCHS=1 \
MAX_STEPS=1 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
PER_DEVICE_EVAL_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=1 \
EVALUATION_STRATEGY=steps \
SAVE_STRATEGY=steps \
EVAL_STEPS=1 \
SAVE_STEPS=1 \
SAVE_TOTAL_LIMIT=1 \
LOGGING_STEPS=1 \
DATALOADER_NUM_WORKERS=0 \
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-0}" \
bash scripts/geometry/train_stage2.sh


echo "Stage 2 smoke test completed."
echo "Trainer state: ${SMOKE_OUTPUT_DIR}/trainer_state.json"
echo "Model config: ${SMOKE_OUTPUT_DIR}/config.json"
echo "Intermediate checkpoint: ${SMOKE_OUTPUT_DIR}/checkpoint-1"
echo "Final model directory: ${SMOKE_OUTPUT_DIR}"