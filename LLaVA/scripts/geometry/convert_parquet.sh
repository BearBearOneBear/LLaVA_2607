#!/bin/bash

set -euo pipefail


# ============================================================
# Stage 1 and Stage 2 Parquet Conversion
#
# 각 stage의 Parquet 파일을 읽어 다음 파일을 생성
#
#   train.json
#   validation.json
#   images/train/
#   images/validation/
#
# LLaVA 저장소 루트에서 실행
# ============================================================


CONVERTER="${CONVERTER:-./tools/geometry/convert_stage1_parquet.py}"


# Stage 1 경로
STAGE1_DATA_DIR="${STAGE1_DATA_DIR:-./geometry_data/stage1_geometry_grounding}"

# Stage 2 경로
STAGE2_DATA_DIR="${STAGE2_DATA_DIR:-./geometry_data/stage2_geometry_grounding}"


if [[ ! -f "${CONVERTER}" ]]; then
    echo "Converter was not found: ${CONVERTER}" >&2
    exit 1
fi


for DATA_DIR in "${STAGE1_DATA_DIR}" "${STAGE2_DATA_DIR}"; do
    if [[ ! -d "${DATA_DIR}/train_parquet" ]]; then
        echo "Train Parquet directory was not found: ${DATA_DIR}/train_parquet" >&2
        exit 1
    fi

    if [[ ! -d "${DATA_DIR}/validation_parquet" ]]; then
        echo "Validation Parquet directory was not found: ${DATA_DIR}/validation_parquet" >&2
        exit 1
    fi
done


# ------------------------------------------------------------
# Stage 1 변환
# ------------------------------------------------------------

echo "Starting Stage 1 dataset conversion."

python "${CONVERTER}" \
    --input_root "${STAGE1_DATA_DIR}" \
    --output_dir "${STAGE1_DATA_DIR}"

echo "Stage 1 dataset conversion completed."


# ------------------------------------------------------------
# Stage 2 변환
# ------------------------------------------------------------

echo "Starting Stage 2 dataset conversion."

python "${CONVERTER}" \
    --input_root "${STAGE2_DATA_DIR}" \
    --output_dir "${STAGE2_DATA_DIR}"

echo "Stage 2 dataset conversion completed."


# ------------------------------------------------------------
# 출력 결과 확인
# ------------------------------------------------------------

for DATA_DIR in "${STAGE1_DATA_DIR}" "${STAGE2_DATA_DIR}"; do
    if [[ ! -f "${DATA_DIR}/train.json" ]]; then
        echo "Converted train JSON was not found: ${DATA_DIR}/train.json" >&2
        exit 1
    fi

    if [[ ! -f "${DATA_DIR}/validation.json" ]]; then
        echo "Converted validation JSON was not found: ${DATA_DIR}/validation.json" >&2
        exit 1
    fi

    if [[ ! -d "${DATA_DIR}/images/train" ]]; then
        echo "Converted train image directory was not found: ${DATA_DIR}/images/train" >&2
        exit 1
    fi

    if [[ ! -d "${DATA_DIR}/images/validation" ]]; then
        echo "Converted validation image directory was not found: ${DATA_DIR}/images/validation" >&2
        exit 1
    fi
done


echo "Stage 1 and Stage 2 dataset conversion completed successfully."
echo "Stage 1 output: ${STAGE1_DATA_DIR}"
echo "Stage 2 output: ${STAGE2_DATA_DIR}"