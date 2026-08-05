#!/bin/bash

set -euo pipefail


# ============================================================
# Stage 1 and Stage 2 Parquet Conversion
#
# Stage 1:
#   train_parquet
#   -> train.json
#   -> images/train
#
# Stage 2:
#   train_parquet + validation_parquet
#   -> train.json + validation.json
#   -> images/train + images/validation
# ============================================================


CONVERTER="${CONVERTER:-./tools/geometry/convert_geometry_parquet.py}"

STAGE1_DATA_DIR="${STAGE1_DATA_DIR:-./geometry_data/stage1_geometry_grounding}"
STAGE2_DATA_DIR="${STAGE2_DATA_DIR:-./geometry_data/stage2_geometry_grounding}"


if [[ ! -f "${CONVERTER}" ]]; then
    echo "Converter was not found: ${CONVERTER}" >&2
    exit 1
fi


# ------------------------------------------------------------
# 입력 데이터 확인
# ------------------------------------------------------------

if [[ ! -d "${STAGE1_DATA_DIR}/train_parquet" ]]; then
    echo "Stage 1 train Parquet directory was not found:" >&2
    echo "${STAGE1_DATA_DIR}/train_parquet" >&2
    exit 1
fi

if [[ ! -d "${STAGE2_DATA_DIR}/train_parquet" ]]; then
    echo "Stage 2 train Parquet directory was not found:" >&2
    echo "${STAGE2_DATA_DIR}/train_parquet" >&2
    exit 1
fi

if [[ ! -d "${STAGE2_DATA_DIR}/validation_parquet" ]]; then
    echo "Stage 2 validation Parquet directory was not found:" >&2
    echo "${STAGE2_DATA_DIR}/validation_parquet" >&2
    exit 1
fi


# ------------------------------------------------------------
# Stage 1 변환: train only
# ------------------------------------------------------------

echo "Starting Stage 1 dataset conversion."

python "${CONVERTER}" \
    --input_root "${STAGE1_DATA_DIR}" \
    --output_dir "${STAGE1_DATA_DIR}" \
    --splits train

echo "Stage 1 dataset conversion completed."


# ------------------------------------------------------------
# Stage 2 변환: train + validation
# ------------------------------------------------------------

echo "Starting Stage 2 dataset conversion."

python "${CONVERTER}" \
    --input_root "${STAGE2_DATA_DIR}" \
    --output_dir "${STAGE2_DATA_DIR}" \
    --splits train validation

echo "Stage 2 dataset conversion completed."


# ------------------------------------------------------------
# Stage 1 출력 확인
# ------------------------------------------------------------

if [[ ! -f "${STAGE1_DATA_DIR}/train.json" ]]; then
    echo "Stage 1 train JSON was not found:" >&2
    echo "${STAGE1_DATA_DIR}/train.json" >&2
    exit 1
fi

if [[ ! -d "${STAGE1_DATA_DIR}/images/train" ]]; then
    echo "Stage 1 train image directory was not found:" >&2
    echo "${STAGE1_DATA_DIR}/images/train" >&2
    exit 1
fi


# ------------------------------------------------------------
# Stage 2 출력 확인
# ------------------------------------------------------------

for FILE_PATH in \
    "${STAGE2_DATA_DIR}/train.json" \
    "${STAGE2_DATA_DIR}/validation.json"
do
    if [[ ! -f "${FILE_PATH}" ]]; then
        echo "Converted Stage 2 JSON was not found: ${FILE_PATH}" >&2
        exit 1
    fi
done

for DIR_PATH in \
    "${STAGE2_DATA_DIR}/images/train" \
    "${STAGE2_DATA_DIR}/images/validation"
do
    if [[ ! -d "${DIR_PATH}" ]]; then
        echo "Converted Stage 2 image directory was not found: ${DIR_PATH}" >&2
        exit 1
    fi
done


echo "Stage 1 and Stage 2 dataset conversion completed successfully."
echo "Stage 1 output: ${STAGE1_DATA_DIR}"
echo "Stage 2 output: ${STAGE2_DATA_DIR}"