#!/bin/bash

set -euo pipefail


# ============================================================
# Stage 1 Smoke Test
#
# 확인 항목:
# - 모델 로드
# - train.json 로드
# - 이미지 로드
# - forward
# - loss
# - backward
# - optimizer step
# - 최종 mm_projector.bin 저장
# ============================================================


# ------------------------------------------------------------
# 데이터 경로
# ------------------------------------------------------------

TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-./geometry_data/stage1_geometry_grounding/train.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-./geometry_data/stage1_geometry_grounding/images}"


# ------------------------------------------------------------
# 출력 경로
# ------------------------------------------------------------

SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-./checkpoints/geometry_stage1_smoke}"


# ------------------------------------------------------------
# 입력 확인
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


rm -rf "${SMOKE_OUTPUT_DIR}"
mkdir -p "${SMOKE_OUTPUT_DIR}"


echo "Starting Stage 1 smoke test."
echo "Training data: ${TRAIN_DATA_PATH}"
echo "Image folder: ${IMAGE_FOLDER}"
echo "Output directory: ${SMOKE_OUTPUT_DIR}"


# ------------------------------------------------------------
# 기존 Stage 1 학습 스크립트를 2 step만 실행
# ------------------------------------------------------------

TRAIN_DATA_PATH="${TRAIN_DATA_PATH}" \
IMAGE_FOLDER="${IMAGE_FOLDER}" \
OUTPUT_DIR="${SMOKE_OUTPUT_DIR}" \
MAX_STEPS=2 \
NUM_TRAIN_EPOCHS=1 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=1 \
LOGGING_STEPS=1 \
DATALOADER_NUM_WORKERS=0 \
bash scripts/geometry/train_stage1.sh


# ------------------------------------------------------------
# projector 저장 확인
# ------------------------------------------------------------

FINAL_PROJECTOR="${SMOKE_OUTPUT_DIR}/mm_projector.bin"

if [[ ! -f "${FINAL_PROJECTOR}" ]]; then
    echo "Stage 1 smoke test failed." >&2
    echo "The projector was not saved:" >&2
    echo "${FINAL_PROJECTOR}" >&2
    exit 1
fi


echo "Stage 1 smoke test completed."
echo "Trainer state: ${SMOKE_OUTPUT_DIR}/trainer_state.json"
echo "Final projector: ${FINAL_PROJECTOR}"