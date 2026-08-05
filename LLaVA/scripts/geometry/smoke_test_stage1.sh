#!/bin/bash

set -euo pipefail


# ============================================================
# Stage 1 Smoke Test
#
# 확인 항목:
# - 실제 LLaVA 모델 로드
# - attention backend 선택
# - train.json 로드
# - 이미지 로드
# - forward
# - loss
# - backward
# - optimizer step
# - 최종 mm_projector.bin 저장
#
# Stage 1 본 학습 스크립트를 2 optimizer step만 실행한다.
# ============================================================


# ------------------------------------------------------------
# 데이터
# ------------------------------------------------------------

TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-./geometry_data/stage1_geometry_grounding/train.json}"

IMAGE_FOLDER="${IMAGE_FOLDER:-./geometry_data/stage1_geometry_grounding/images}"


# ------------------------------------------------------------
# 출력
# ------------------------------------------------------------

SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-./checkpoints/geometry_stage1_smoke}"


# ------------------------------------------------------------
# 삭제 가드
#
# 다음 조건을 모두 만족해야 삭제한다.
# - 저장소의 checkpoints 디렉터리 아래
# - 마지막 디렉터리 이름에 smoke 포함
#
# 본 학습 결과인 geometry_stage1 등을 실수로 삭제하지 않는다.
# ------------------------------------------------------------

safe_remove_smoke_directory() {
    local TARGET="${1:-}"

    if [[ -z "${TARGET}" ]]; then
        echo "Refusing to remove an empty path." >&2
        exit 1
    fi

    local REPOSITORY_ROOT
    local RESOLVED_TARGET
    local DIRECTORY_NAME

    REPOSITORY_ROOT="$(pwd -P)"

    RESOLVED_TARGET="$(
        python - "${TARGET}" <<'PY'
import os
import sys

print(os.path.realpath(sys.argv[1]))
PY
    )"

    DIRECTORY_NAME="$(basename "${RESOLVED_TARGET}")"

    case "${RESOLVED_TARGET}" in
        "${REPOSITORY_ROOT}/checkpoints/"*)
            ;;
        *)
            echo "Refusing to remove a path outside checkpoints:" >&2
            echo "${RESOLVED_TARGET}" >&2
            exit 1
            ;;
    esac

    case "${DIRECTORY_NAME}" in
        *smoke*)
            ;;
        *)
            echo "Refusing to remove a non-smoke directory:" >&2
            echo "${RESOLVED_TARGET}" >&2
            echo "The directory name must contain 'smoke'." >&2
            exit 1
            ;;
    esac

    if [[ -e "${RESOLVED_TARGET}" ]]; then
        rm -rf -- "${RESOLVED_TARGET}"
    fi
}


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

if [[ ! -f "scripts/geometry/train_stage1.sh" ]]; then
    echo "Stage 1 training script was not found:" >&2
    echo "scripts/geometry/train_stage1.sh" >&2
    exit 1
fi


# ------------------------------------------------------------
# 이전 smoke output 제거
# ------------------------------------------------------------

safe_remove_smoke_directory "${SMOKE_OUTPUT_DIR}"
mkdir -p "${SMOKE_OUTPUT_DIR}"


# ------------------------------------------------------------
# 설정 출력
# ------------------------------------------------------------

echo "Starting Stage 1 smoke test."
echo "Training data: ${TRAIN_DATA_PATH}"
echo "Image folder: ${IMAGE_FOLDER}"
echo "Output directory: ${SMOKE_OUTPUT_DIR}"
echo "Attention implementation: ${ATTENTION_IMPLEMENTATION:-auto}"


# ------------------------------------------------------------
# Stage 1 본 학습 스크립트를 2 step만 실행
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
ATTENTION_IMPLEMENTATION="${ATTENTION_IMPLEMENTATION:-auto}" \
bash scripts/geometry/train_stage1.sh


# ------------------------------------------------------------
# 결과 확인
# ------------------------------------------------------------

FINAL_PROJECTOR="${SMOKE_OUTPUT_DIR}/mm_projector.bin"

if [[ ! -f "${FINAL_PROJECTOR}" ]]; then
    echo "Stage 1 smoke test failed." >&2
    echo "The projector was not saved:" >&2
    echo "${FINAL_PROJECTOR}" >&2
    exit 1
fi


echo "Stage 1 smoke test completed successfully."
echo "Final projector: ${FINAL_PROJECTOR}"

if [[ -f "${SMOKE_OUTPUT_DIR}/trainer_state.json" ]]; then
    echo "Trainer state: ${SMOKE_OUTPUT_DIR}/trainer_state.json"
fi