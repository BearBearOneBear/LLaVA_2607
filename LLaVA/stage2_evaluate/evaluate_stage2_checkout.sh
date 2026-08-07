#!/bin/bash
set -euo pipefail

# ============================================================
# Stage-2 Checkpoint Evaluator
#
# Directory layout expected:
#   LLaVA/
#   ├── checkpoints/geometry_stage1
#   ├── checkpoints/geometry_stage2
#   ├── stage2_test_data
#   └── stage2_evaludate/
#       ├── inspect_artifacts.py
#       ├── inspect_weights.py
#       ├── evaluate_behavior.py
#       ├── evaluate_language.py
#       ├── summarize.py
#       └── evaluate_stage2_checkout.sh
#
# Layer 0 : integrity + weight delta
# Layer 1 : Stage1/Stage2 behavior
# Layer 2 : representation audit (deferred)
# Layer 3 : lightweight language preservation
# Layer 4 : Stage3 transfer stress tests
# ============================================================

is_true() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    true|1|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BASE_MODEL="${BASE_MODEL:-liuhaotian/llava-v1.5-7b}"
STAGE1_DIR="${STAGE1_DIR:-${REPO_ROOT}/checkpoints/geometry_stage1}"
STAGE2_DIR="${STAGE2_DIR:-${REPO_ROOT}/checkpoints/geometry_stage2}"
TEST_DATA_DIR="${TEST_DATA_DIR:-${REPO_ROOT}/stage2_test_data}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results}"
LOGS_ROOT="${LOGS_ROOT:-${REPO_ROOT}/logs/geometry_pipeline}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

RUN_INTEGRITY="${RUN_INTEGRITY:-True}"
RUN_WEIGHT_AUDIT="${RUN_WEIGHT_AUDIT:-True}"
RUN_BEHAVIOR="${RUN_BEHAVIOR:-True}"
RUN_LANGUAGE="${RUN_LANGUAGE:-True}"
RUN_STAGE3_TRANSFER="${RUN_STAGE3_TRANSFER:-True}"
RUN_SUMMARY="${RUN_SUMMARY:-True}"

# 0 means all rows. Example: MAX_SAMPLES=20 for smoke test.
MAX_SAMPLES="${MAX_SAMPLES:-0}"
RESUME="${RESUME:-True}"

# Expensive LLM truncated SVD is disabled by default.
SVD_TOPK="${SVD_TOPK:-0}"

# WikiText requires Hugging Face network/cache access.
SKIP_PPL="${SKIP_PPL:-False}"
PPL_MAX_CHARS="${PPL_MAX_CHARS:-1500000}"

required_files=(
  "${SCRIPT_DIR}/inspect_artifacts.py"
  "${SCRIPT_DIR}/inspect_weights.py"
  "${SCRIPT_DIR}/evaluate_behavior.py"
  "${SCRIPT_DIR}/evaluate_language.py"
  "${SCRIPT_DIR}/summarize.py"
)
for file in "${required_files[@]}"; do
  if [[ ! -f "${file}" ]]; then
    echo "Required evaluator file was not found: ${file}" >&2
    exit 1
  fi
done

if [[ ! -d "${TEST_DATA_DIR}" ]]; then
  echo "Test data directory was not found: ${TEST_DATA_DIR}" >&2
  exit 1
fi
if [[ ! -f "${STAGE1_DIR}/mm_projector.bin" ]]; then
  echo "Stage-1 projector was not found: ${STAGE1_DIR}/mm_projector.bin" >&2
  exit 1
fi
if [[ ! -f "${STAGE2_DIR}/config.json" ]]; then
  echo "Stage-2 config was not found: ${STAGE2_DIR}/config.json" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
RUN_LOG="${OUTPUT_DIR}/evaluate_stage2_checkout.log"
exec > >(tee "${RUN_LOG}") 2>&1

resume_arg=()
if is_true "${RESUME}"; then
  resume_arg+=(--resume)
fi

max_arg=()
if [[ "${MAX_SAMPLES}" != "0" ]]; then
  max_arg+=(--max-samples "${MAX_SAMPLES}")
fi

echo "============================================================"
echo "Stage-2 checkpoint evaluation"
echo "Repository root: ${REPO_ROOT}"
echo "Evaluator directory: ${SCRIPT_DIR}"
echo "Base model: ${BASE_MODEL}"
echo "Stage-1 directory: ${STAGE1_DIR}"
echo "Stage-2 directory: ${STAGE2_DIR}"
echo "Test data: ${TEST_DATA_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Run log: ${RUN_LOG}"
echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Max samples per dataset: ${MAX_SAMPLES}"
echo "Representation audit: deferred"
echo "============================================================"

if is_true "${RUN_INTEGRITY}"; then
  echo "[0A] Inspecting saved training artifacts."
  python "${SCRIPT_DIR}/inspect_artifacts.py" \
    --stage1-dir "${STAGE1_DIR}" \
    --stage2-dir "${STAGE2_DIR}" \
    --logs-root "${LOGS_ROOT}" \
    --output "${OUTPUT_DIR}/integrity.json"
else
  echo "[0A] Integrity inspection skipped."
fi

if is_true "${RUN_WEIGHT_AUDIT}"; then
  echo "[0B] Analyzing Stage-2 weight changes on CPU."
  python "${SCRIPT_DIR}/inspect_weights.py" \
    --base-model "${BASE_MODEL}" \
    --stage1-dir "${STAGE1_DIR}" \
    --stage2-dir "${STAGE2_DIR}" \
    --output-dir "${OUTPUT_DIR}/weight_audit" \
    --svd-topk "${SVD_TOPK}"
else
  echo "[0B] Weight audit skipped."
fi

if is_true "${RUN_BEHAVIOR}"; then
  for model_kind in base stage1 stage2; do
    echo "[1] Behavior evaluation: ${model_kind}"
    python "${SCRIPT_DIR}/evaluate_behavior.py" \
      --model-kind "${model_kind}" \
      --base-model "${BASE_MODEL}" \
      --stage1-dir "${STAGE1_DIR}" \
      --stage2-dir "${STAGE2_DIR}" \
      --data-dir "${TEST_DATA_DIR}" \
      --datasets stage1 stage2 \
      --output-dir "${OUTPUT_DIR}" \
      "${resume_arg[@]}" \
      "${max_arg[@]}"
  done
else
  echo "[1] Behavior evaluation skipped."
fi

echo "[2] Representation audit deferred by design."

if is_true "${RUN_LANGUAGE}"; then
  ppl_arg=()
  if is_true "${SKIP_PPL}"; then
    ppl_arg+=(--skip-ppl)
  fi
  for model_kind in base stage2; do
    echo "[3] Language preservation: ${model_kind}"
    python "${SCRIPT_DIR}/evaluate_language.py" \
      --model-kind "${model_kind}" \
      --base-model "${BASE_MODEL}" \
      --stage2-dir "${STAGE2_DIR}" \
      --output-dir "${OUTPUT_DIR}/language" \
      --ppl-max-chars "${PPL_MAX_CHARS}" \
      "${ppl_arg[@]}"
  done
else
  echo "[3] Language preservation skipped."
fi

if is_true "${RUN_STAGE3_TRANSFER}"; then
  echo "[4] Stage3 transfer stress tests."
  python "${SCRIPT_DIR}/evaluate_behavior.py" \
    --model-kind stage2 \
    --base-model "${BASE_MODEL}" \
    --stage1-dir "${STAGE1_DIR}" \
    --stage2-dir "${STAGE2_DIR}" \
    --data-dir "${TEST_DATA_DIR}" \
    --datasets stage3_base stage3_values stage3_unseen stage3_wide \
    --output-dir "${OUTPUT_DIR}" \
    "${resume_arg[@]}" \
    "${max_arg[@]}"
else
  echo "[4] Stage3 transfer evaluation skipped."
fi

if is_true "${RUN_SUMMARY}"; then
  echo "[Summary] Combining metrics."
  python "${SCRIPT_DIR}/summarize.py" --root "${OUTPUT_DIR}"
else
  echo "[Summary] Summary generation skipped."
fi

echo "============================================================"
echo "Stage-2 checkpoint evaluation completed."
echo "Results directory: ${OUTPUT_DIR}"
echo "Main report: ${OUTPUT_DIR}/evaluation_summary.md"
echo "Machine-readable summary: ${OUTPUT_DIR}/evaluation_summary.json"
echo "Full console log: ${RUN_LOG}"
echo "============================================================"
