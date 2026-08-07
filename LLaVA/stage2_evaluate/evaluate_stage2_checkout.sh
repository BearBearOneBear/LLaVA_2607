#!/bin/bash
set -uo pipefail

# ============================================================
# Stage-2 Checkpoint Evaluator
#
# Fail-open policy:
#   - Each evaluation step runs independently.
#   - A failed evaluator is recorded and the next evaluator is attempted.
#   - If a Python process dies from CUDA OOM but the shell survives, later
#     evaluations are still attempted.
#   - If the whole job/shell is killed by the OS, scheduler, or GPU driver,
#     no shell script can continue after that point.
#
# Expected layout:
#   LLaVA/
#   ├── checkpoints/geometry_stage1
#   ├── checkpoints/geometry_stage2
#   ├── stage2_test_data
#   └── stage2_evaluate/
#       ├── inspect_artifacts.py
#       ├── inspect_weights.py
#       ├── evaluate_behavior.py
#       ├── evaluate_representation.py
#       ├── evaluate_language.py
#       ├── summarize.py
#       └── evaluate_stage2_checkout.sh
#
# Layer 0 : integrity + weight delta
# Layer 1 : Stage1/Stage2 behavior + visual ablations
# Layer 2 : representation anchor sanity check for Stage3 KD
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
VICUNA_MODEL="${VICUNA_MODEL:-lmsys/vicuna-7b-v1.5}"
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
RUN_REPRESENTATION="${RUN_REPRESENTATION:-True}"
RUN_LANGUAGE="${RUN_LANGUAGE:-True}"
RUN_STAGE3_TRANSFER="${RUN_STAGE3_TRANSFER:-True}"
RUN_IMAGE_ABLATIONS="${RUN_IMAGE_ABLATIONS:-True}"
RUN_STAGE3_IMAGE_ABLATIONS="${RUN_STAGE3_IMAGE_ABLATIONS:-True}"
IMAGE_ABLATION_MODES="${IMAGE_ABLATION_MODES:-shuffled blank none}"
RUN_SUMMARY="${RUN_SUMMARY:-True}"

# 0 means all rows. Example: MAX_SAMPLES=20 for a smoke test.
MAX_SAMPLES="${MAX_SAMPLES:-0}"
RESUME="${RESUME:-True}"

# Weight audit options.
SVD_TOPK="${SVD_TOPK:-0}"

# Representation audit options.
REP_LAYERS="${REP_LAYERS:-8 16 24 32}"
REP_STAGE2_MAX_SAMPLES="${REP_STAGE2_MAX_SAMPLES:-0}"
REP_STAGE3_SAMPLES="${REP_STAGE3_SAMPLES:-200}"

# Language audit options.
SKIP_PPL="${SKIP_PPL:-False}"
PPL_MAX_CHARS="${PPL_MAX_CHARS:-1500000}"

if ! mkdir -p "${OUTPUT_DIR}"; then
  echo "Cannot create output directory: ${OUTPUT_DIR}" >&2
  exit 1
fi
STEP_LOG_DIR="${OUTPUT_DIR}/step_logs"
mkdir -p "${STEP_LOG_DIR}"
STATUS_FILE="${OUTPUT_DIR}/step_status.tsv"
RUN_LOG="${OUTPUT_DIR}/evaluate_stage2_checkout.log"

printf 'step\tstatus\texit_code\tlog\n' > "${STATUS_FILE}"
exec > >(tee "${RUN_LOG}") 2>&1

record_status() {
  local name="$1"
  local status="$2"
  local code="$3"
  local log_path="${4:-}"
  printf '%s\t%s\t%s\t%s\n' "${name}" "${status}" "${code}" "${log_path}" >> "${STATUS_FILE}"
}

record_skipped() {
  local name="$1"
  local reason="${2:-disabled}"
  echo "SKIPPED: ${name} (${reason})"
  record_status "${name}" "SKIPPED" "-" "${reason}"
}

LAST_STEP_RC=0
run_step() {
  local name="$1"
  shift
  local safe_name
  safe_name="$(printf '%s' "${name}" | tr '/ :+' '_____')"
  local step_log="${STEP_LOG_DIR}/${safe_name}.log"

  echo
  echo "============================================================"
  echo "START: ${name}"
  echo "============================================================"

  "$@" 2>&1 | tee "${step_log}"
  local rc=${PIPESTATUS[0]}
  LAST_STEP_RC=${rc}

  if [[ ${rc} -eq 0 ]]; then
    echo "SUCCESS: ${name}"
    record_status "${name}" "SUCCESS" "0" "${step_log}"
  else
    echo "WARNING: ${name} failed with exit code ${rc}." >&2
    echo "Continuing with the next evaluation." >&2
    record_status "${name}" "FAILED" "${rc}" "${step_log}"
  fi

  # Always return success so one evaluator cannot stop the orchestration.
  return 0
}

warn_if_missing() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "WARNING: ${label} was not found: ${path}" >&2
    echo "         Dependent steps may fail, but independent steps will still run." >&2
  fi
}

# Preflight is warning-only so that as many independent results as possible survive.
warn_if_missing "${SCRIPT_DIR}/inspect_artifacts.py" "inspect_artifacts.py"
warn_if_missing "${SCRIPT_DIR}/inspect_weights.py" "inspect_weights.py"
warn_if_missing "${SCRIPT_DIR}/evaluate_behavior.py" "evaluate_behavior.py"
warn_if_missing "${SCRIPT_DIR}/evaluate_representation.py" "evaluate_representation.py"
warn_if_missing "${SCRIPT_DIR}/evaluate_language.py" "evaluate_language.py"
warn_if_missing "${SCRIPT_DIR}/summarize.py" "summarize.py"
warn_if_missing "${TEST_DATA_DIR}" "test data directory"
warn_if_missing "${STAGE1_DIR}/mm_projector.bin" "Stage-1 projector"
warn_if_missing "${STAGE2_DIR}/config.json" "Stage-2 config"

resume_arg=()
if is_true "${RESUME}"; then
  resume_arg+=(--resume)
fi

max_arg=()
if [[ "${MAX_SAMPLES}" != "0" ]]; then
  max_arg+=(--max-samples "${MAX_SAMPLES}")
fi

# A smoke-test MAX_SAMPLES should also cap representation work.
rep_stage2_samples="${REP_STAGE2_MAX_SAMPLES}"
rep_stage3_samples="${REP_STAGE3_SAMPLES}"
if [[ "${MAX_SAMPLES}" != "0" ]]; then
  if [[ "${rep_stage2_samples}" == "0" ]] || (( rep_stage2_samples > MAX_SAMPLES )); then
    rep_stage2_samples="${MAX_SAMPLES}"
  fi
  if (( rep_stage3_samples > MAX_SAMPLES )); then
    rep_stage3_samples="${MAX_SAMPLES}"
  fi
fi
read -r -a rep_layer_args <<< "${REP_LAYERS}"

echo "============================================================"
echo "Stage-2 checkpoint evaluation"
echo "Repository root: ${REPO_ROOT}"
echo "Evaluator directory: ${SCRIPT_DIR}"
echo "Base model: ${BASE_MODEL}"
echo "Vicuna student candidate: ${VICUNA_MODEL}"
echo "Stage-1 directory: ${STAGE1_DIR}"
echo "Stage-2 directory: ${STAGE2_DIR}"
echo "Test data: ${TEST_DATA_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Run log: ${RUN_LOG}"
echo "Status file: ${STATUS_FILE}"
echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Max samples per dataset: ${MAX_SAMPLES}"
echo "Representation audit: ${RUN_REPRESENTATION}"
echo "Representation layers: ${REP_LAYERS}"
echo "Representation Stage2 samples: ${rep_stage2_samples}"
echo "Representation Stage3-base samples: ${rep_stage3_samples}"
echo "Image ablations: ${RUN_IMAGE_ABLATIONS} (${IMAGE_ABLATION_MODES})"
echo "Stage3 image ablations: ${RUN_STAGE3_IMAGE_ABLATIONS}"
echo "Failure policy: fail-open per evaluator"
echo "============================================================"

# ---------------------------------------------------------------------------
# Layer 0A: saved artifacts / trainer state
# ---------------------------------------------------------------------------
if is_true "${RUN_INTEGRITY}"; then
  run_step "integrity" \
    python "${SCRIPT_DIR}/inspect_artifacts.py" \
      --stage1-dir "${STAGE1_DIR}" \
      --stage2-dir "${STAGE2_DIR}" \
      --logs-root "${LOGS_ROOT}" \
      --output "${OUTPUT_DIR}/integrity.json"
else
  record_skipped "integrity" "RUN_INTEGRITY=False"
fi

# ---------------------------------------------------------------------------
# Layer 0B: Stage1->Stage2 projector and Base->Stage2 LLM weight changes
# ---------------------------------------------------------------------------
if is_true "${RUN_WEIGHT_AUDIT}"; then
  run_step "weight_audit" \
    python "${SCRIPT_DIR}/inspect_weights.py" \
      --base-model "${BASE_MODEL}" \
      --stage1-dir "${STAGE1_DIR}" \
      --stage2-dir "${STAGE2_DIR}" \
      --output-dir "${OUTPUT_DIR}/weight_audit" \
      --svd-topk "${SVD_TOPK}"
else
  record_skipped "weight_audit" "RUN_WEIGHT_AUDIT=False"
fi

# ---------------------------------------------------------------------------
# Layer 1: Stage1 / Stage2 behavior
# ---------------------------------------------------------------------------
if is_true "${RUN_BEHAVIOR}"; then
  for model_kind in base stage1 stage2; do
    run_step "behavior_${model_kind}_normal" \
      python "${SCRIPT_DIR}/evaluate_behavior.py" \
        --model-kind "${model_kind}" \
        --base-model "${BASE_MODEL}" \
        --stage1-dir "${STAGE1_DIR}" \
        --stage2-dir "${STAGE2_DIR}" \
        --data-dir "${TEST_DATA_DIR}" \
        --datasets stage1 stage2 \
        --image-mode normal \
        --output-dir "${OUTPUT_DIR}" \
        "${resume_arg[@]}" \
        "${max_arg[@]}"
  done

  if is_true "${RUN_IMAGE_ABLATIONS}"; then
    for image_mode in ${IMAGE_ABLATION_MODES}; do
      run_step "behavior_stage2_${image_mode}" \
        python "${SCRIPT_DIR}/evaluate_behavior.py" \
          --model-kind stage2 \
          --base-model "${BASE_MODEL}" \
          --stage1-dir "${STAGE1_DIR}" \
          --stage2-dir "${STAGE2_DIR}" \
          --data-dir "${TEST_DATA_DIR}" \
          --datasets stage1 stage2 \
          --image-mode "${image_mode}" \
          --output-dir "${OUTPUT_DIR}" \
          "${resume_arg[@]}" \
          "${max_arg[@]}"
    done
  else
    for image_mode in ${IMAGE_ABLATION_MODES}; do
      record_skipped "behavior_stage2_${image_mode}" "RUN_IMAGE_ABLATIONS=False"
    done
  fi
else
  record_skipped "behavior_base_normal" "RUN_BEHAVIOR=False"
  record_skipped "behavior_stage1_normal" "RUN_BEHAVIOR=False"
  record_skipped "behavior_stage2_normal" "RUN_BEHAVIOR=False"
  for image_mode in ${IMAGE_ABLATION_MODES}; do
    record_skipped "behavior_stage2_${image_mode}" "RUN_BEHAVIOR=False"
  done
fi

# ---------------------------------------------------------------------------
# Layer 2: representation anchor sanity check for Stage3 KD design
# ---------------------------------------------------------------------------
if is_true "${RUN_REPRESENTATION}"; then
  run_step "representation" \
    python "${SCRIPT_DIR}/evaluate_representation.py" \
      --base-model "${BASE_MODEL}" \
      --vicuna-model "${VICUNA_MODEL}" \
      --stage2-dir "${STAGE2_DIR}" \
      --data-dir "${TEST_DATA_DIR}" \
      --output-dir "${OUTPUT_DIR}/representation" \
      --layers "${rep_layer_args[@]}" \
      --stage2-max-samples "${rep_stage2_samples}" \
      --stage3-samples "${rep_stage3_samples}"
else
  record_skipped "representation" "RUN_REPRESENTATION=False"
fi

# ---------------------------------------------------------------------------
# Layer 3: lightweight language preservation
# ---------------------------------------------------------------------------
if is_true "${RUN_LANGUAGE}"; then
  ppl_arg=()
  if is_true "${SKIP_PPL}"; then
    ppl_arg+=(--skip-ppl)
  fi
  for model_kind in base stage2; do
    run_step "language_${model_kind}" \
      python "${SCRIPT_DIR}/evaluate_language.py" \
        --model-kind "${model_kind}" \
        --base-model "${BASE_MODEL}" \
        --stage2-dir "${STAGE2_DIR}" \
        --output-dir "${OUTPUT_DIR}/language" \
        --ppl-max-chars "${PPL_MAX_CHARS}" \
        "${ppl_arg[@]}"
  done
else
  record_skipped "language_base" "RUN_LANGUAGE=False"
  record_skipped "language_stage2" "RUN_LANGUAGE=False"
fi

# ---------------------------------------------------------------------------
# Layer 4: Stage3 transfer stress tests
# ---------------------------------------------------------------------------
if is_true "${RUN_STAGE3_TRANSFER}"; then
  run_step "stage3_normal" \
    python "${SCRIPT_DIR}/evaluate_behavior.py" \
      --model-kind stage2 \
      --base-model "${BASE_MODEL}" \
      --stage1-dir "${STAGE1_DIR}" \
      --stage2-dir "${STAGE2_DIR}" \
      --data-dir "${TEST_DATA_DIR}" \
      --datasets stage3_base stage3_values stage3_unseen stage3_wide \
      --image-mode normal \
      --output-dir "${OUTPUT_DIR}" \
      "${resume_arg[@]}" \
      "${max_arg[@]}"

  if is_true "${RUN_STAGE3_IMAGE_ABLATIONS}"; then
    for image_mode in ${IMAGE_ABLATION_MODES}; do
      run_step "stage3_${image_mode}" \
        python "${SCRIPT_DIR}/evaluate_behavior.py" \
          --model-kind stage2 \
          --base-model "${BASE_MODEL}" \
          --stage1-dir "${STAGE1_DIR}" \
          --stage2-dir "${STAGE2_DIR}" \
          --data-dir "${TEST_DATA_DIR}" \
          --datasets stage3_base stage3_values stage3_unseen stage3_wide \
          --image-mode "${image_mode}" \
          --output-dir "${OUTPUT_DIR}" \
          "${resume_arg[@]}" \
          "${max_arg[@]}"
    done
  else
    for image_mode in ${IMAGE_ABLATION_MODES}; do
      record_skipped "stage3_${image_mode}" "RUN_STAGE3_IMAGE_ABLATIONS=False"
    done
  fi
else
  record_skipped "stage3_normal" "RUN_STAGE3_TRANSFER=False"
  for image_mode in ${IMAGE_ABLATION_MODES}; do
    record_skipped "stage3_${image_mode}" "RUN_STAGE3_TRANSFER=False"
  done
fi

write_fallback_summary() {
  local summary_path="${OUTPUT_DIR}/evaluation_summary.md"
  {
    echo "# Stage-2 Checkpoint Evaluation Summary"
    echo
    echo "> WARNING: summarize.py failed. This fallback report contains orchestration status only."
    echo "> Raw successful outputs remain under: ${OUTPUT_DIR}"
    echo
    echo "## Execution status"
    echo
    echo '| step | status | exit code | log |'
    echo '|---|---|---:|---|'
    tail -n +2 "${STATUS_FILE}" | while IFS=$'\t' read -r step status code log_path; do
      printf '| %s | %s | %s | %s |\n' "${step}" "${status}" "${code}" "${log_path}"
    done
    echo
    echo "Full console log: ${RUN_LOG}"
  } > "${summary_path}"
}

# ---------------------------------------------------------------------------
# Final summary. Missing/failed upstream outputs are intentionally tolerated by
# summarize.py; they appear as N/A together with the execution-status table.
# ---------------------------------------------------------------------------
if is_true "${RUN_SUMMARY}"; then
  run_step "summary_generation" \
    python "${SCRIPT_DIR}/summarize.py" --root "${OUTPUT_DIR}"
  summary_rc=${LAST_STEP_RC}
  if [[ ${summary_rc} -ne 0 || ! -s "${OUTPUT_DIR}/evaluation_summary.md" ]]; then
    echo "WARNING: summarize.py did not produce a usable report; writing fallback summary." >&2
    write_fallback_summary
  fi
else
  record_skipped "summary_generation" "RUN_SUMMARY=False"
  write_fallback_summary
fi

echo
echo "============================================================"
echo "Stage-2 checkpoint evaluation finished (fail-open mode)."
echo "Results directory: ${OUTPUT_DIR}"
echo "Main report to download: ${OUTPUT_DIR}/evaluation_summary.md"
echo "Machine-readable summary: ${OUTPUT_DIR}/evaluation_summary.json"
echo "Execution status: ${STATUS_FILE}"
echo "Full console log: ${RUN_LOG}"
echo "============================================================"

# Deliberately return 0 if the shell itself survived. Individual failures are
# encoded in evaluation_summary.md and step_status.tsv.
exit 0
