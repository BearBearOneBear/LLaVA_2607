#!/bin/bash

set -euo pipefail


# ============================================================
# Stage 1 and Stage 2 Training Pipeline
#
#   1. Stage 1/2 Parquet 변환
#   2. Stage 1 smoke test
#   3. Stage 1 projector 학습
#   4. Stage 2 smoke test
#   5. Stage 2 LLM + projector 학습
#
# 중간 단계에서 오류가 발생하면 즉시 중단한다.
# LLaVA 저장소 루트에서 실행한다.
# ============================================================


# ------------------------------------------------------------
# 실행 여부 설정
#
# True  : 해당 단계 실행
# False : 해당 단계 생략
# ------------------------------------------------------------

RUN_DATA_CONVERSION="${RUN_DATA_CONVERSION:-True}"
RUN_STAGE1_SMOKE="${RUN_STAGE1_SMOKE:-True}"
RUN_STAGE1_TRAINING="${RUN_STAGE1_TRAINING:-True}"
RUN_STAGE2_SMOKE="${RUN_STAGE2_SMOKE:-True}"
RUN_STAGE2_TRAINING="${RUN_STAGE2_TRAINING:-True}"


# ------------------------------------------------------------
# 출력 경로
# ------------------------------------------------------------

STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-./checkpoints/geometry_stage1}"
STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-./checkpoints/geometry_stage2}"

# Stage 1은 validation 없이 1 epoch을 실행하므로
# 최종 output directory에 저장된 projector를 직접 사용한다.
STAGE1_PROJECTOR_PATH="${STAGE1_PROJECTOR_PATH:-${STAGE1_OUTPUT_DIR}/mm_projector.bin}"


# ------------------------------------------------------------
# 로그 경로
# ------------------------------------------------------------

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-./logs/geometry_pipeline/${RUN_ID}}"

mkdir -p "${LOG_DIR}"


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
# 필요한 실행 파일 확인
# ------------------------------------------------------------

REQUIRED_FILES=(
    "scripts/geometry/convert_parquet.sh"
    "scripts/geometry/smoke_test_stage1.sh"
    "scripts/geometry/train_stage1.sh"
    "scripts/geometry/smoke_test_stage2.sh"
    "scripts/geometry/train_stage2.sh"
)

for FILE_PATH in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "${FILE_PATH}" ]]; then
        echo "Required file was not found: ${FILE_PATH}" >&2
        exit 1
    fi
done


echo "Starting Stage 1 and Stage 2 pipeline."
echo "Log directory: ${LOG_DIR}"
echo "Stage 1 output directory: ${STAGE1_OUTPUT_DIR}"
echo "Stage 2 output directory: ${STAGE2_OUTPUT_DIR}"


# ============================================================
# 1. Parquet 변환
# ============================================================

if is_true "${RUN_DATA_CONVERSION}"; then
    echo "Step 1: Starting dataset conversion."

    bash scripts/geometry/convert_parquet.sh \
        2>&1 | tee "${LOG_DIR}/01_data_conversion.log"

    echo "Step 1: Dataset conversion completed."
else
    echo "Step 1: Dataset conversion skipped."
fi


# ============================================================
# 2. Stage 1 smoke test
# ============================================================

if is_true "${RUN_STAGE1_SMOKE}"; then
    echo "Step 2: Starting Stage 1 smoke test."

    bash scripts/geometry/smoke_test_stage1.sh \
        2>&1 | tee "${LOG_DIR}/02_stage1_smoke.log"

    echo "Step 2: Stage 1 smoke test completed."
else
    echo "Step 2: Stage 1 smoke test skipped."
fi


# ============================================================
# 3. Stage 1 projector 학습
# ============================================================

if is_true "${RUN_STAGE1_TRAINING}"; then
    echo "Step 3: Starting Stage 1 training."

    OUTPUT_DIR="${STAGE1_OUTPUT_DIR}" \
    bash scripts/geometry/train_stage1.sh \
        2>&1 | tee "${LOG_DIR}/03_stage1_training.log"

    echo "Step 3: Stage 1 training completed."
else
    echo "Step 3: Stage 1 training skipped."
fi


# ------------------------------------------------------------
# Stage 2 실행 전 Stage 1 projector 확인
# ------------------------------------------------------------

if is_true "${RUN_STAGE2_SMOKE}" || is_true "${RUN_STAGE2_TRAINING}"; then
    if [[ ! -f "${STAGE1_PROJECTOR_PATH}" ]]; then
        echo "Stage 1 projector was not found:" >&2
        echo "${STAGE1_PROJECTOR_PATH}" >&2
        exit 1
    fi
fi


# ============================================================
# 4. Stage 2 smoke test
# ============================================================

if is_true "${RUN_STAGE2_SMOKE}"; then
    echo "Step 4: Starting Stage 2 smoke test."

    STAGE1_PROJECTOR_PATH="${STAGE1_PROJECTOR_PATH}" \
    bash scripts/geometry/smoke_test_stage2.sh \
        2>&1 | tee "${LOG_DIR}/04_stage2_smoke.log"

    echo "Step 4: Stage 2 smoke test completed."
else
    echo "Step 4: Stage 2 smoke test skipped."
fi


# ============================================================
# 5. Stage 2 학습
# ============================================================

if is_true "${RUN_STAGE2_TRAINING}"; then
    echo "Step 5: Starting Stage 2 training."

    STAGE1_PROJECTOR_PATH="${STAGE1_PROJECTOR_PATH}" \
    OUTPUT_DIR="${STAGE2_OUTPUT_DIR}" \
    bash scripts/geometry/train_stage2.sh \
        2>&1 | tee "${LOG_DIR}/05_stage2_training.log"

    echo "Step 5: Stage 2 training completed."
else
    echo "Step 5: Stage 2 training skipped."
fi


echo "Stage 1 and Stage 2 pipeline completed successfully."
echo "Stage 1 output: ${STAGE1_OUTPUT_DIR}"
echo "Stage 1 projector: ${STAGE1_PROJECTOR_PATH}"
echo "Stage 2 output: ${STAGE2_OUTPUT_DIR}"
echo "Logs: ${LOG_DIR}"