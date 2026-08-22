#!/usr/bin/env bash

# ==============================================================================
# Geometry evaluation pipeline
#
# One evaluator may fail without stopping the remaining evaluators.
#
# The Python runner records each status independently and continues.
# ==============================================================================

set -uo pipefail


# ==============================================================================
# 0. Repository
# ==============================================================================

# LLaVA repository root.
REPO_ROOT="/path/to/LLaVA_2607/LLaVA"

# Folder containing:
# accuracy.py
# anchor.py
# represent.py
# visual_dependency.py
# language.py
# log.py
# helper.py
# ontology.py
# run_all.py
EVAL_DIR="${REPO_ROOT}/geometry_evaluate"


# ==============================================================================
# 1. Base model
# ==============================================================================

BASE_MODEL="liuhaotian/llava-v1.5-7b"


# ==============================================================================
# 2. Checkpoints
# ==============================================================================

# Stage1 projector checkpoint
STAGE1_DIR="${REPO_ROOT}/checkpoints/geometry_stage1"

# Stage1 -> Stage2 curriculum checkpoint
STAGE2_DIR="${REPO_ROOT}/checkpoints/geometry_stage2"

# Base -> Stage2-only control checkpoint
STAGE2_ONLY_DIR="${REPO_ROOT}/checkpoints/geometry_stage2_only"


# ==============================================================================
# 3. Evaluation datasets
# ==============================================================================

# Directory containing:
# stage1_geometry_evaluate_evaluate_*.parquet
STAGE1_DATA_DIR="${REPO_ROOT}/stage2_test_data"

# Directory containing:
# stage2_geometry_evaluate_evaluate_*.parquet
STAGE2_DATA_DIR="${REPO_ROOT}/stage2_test_data"


# ==============================================================================
# 4. Training logs
# ==============================================================================
#
# These may be:
#   - trainer_state.json itself
#   - or a training output/checkpoint directory containing trainer_state.json
#
# log.py searches for the most complete trainer_state.json.
# ==============================================================================

STAGE1_LOG="${STAGE1_DIR}"
STAGE2_LOG="${STAGE2_DIR}"
STAGE2_ONLY_LOG="${STAGE2_ONLY_DIR}"


# ==============================================================================
# 5. Result directory
# ==============================================================================
#
# Download this whole directory after evaluation.
# ==============================================================================

RESULT_DIR="${REPO_ROOT}/geometry_evaluate_result"


# ==============================================================================
# 6. Optional language corpus
# ==============================================================================
#
# Leave empty to let language.py use its WikiText-2 fallback chain.
#
# Example:
# LANGUAGE_TEXT_FILE="/path/to/wikitext2_test.txt"
# ==============================================================================

LANGUAGE_TEXT_FILE=""


# ==============================================================================
# 7. Runtime
# ==============================================================================
#
# 0 = full evaluation.
#
# For a smoke test, use e.g.:
# MAX_SAMPLES=108
#
# NOTE:
# represent.py requires at least 54 samples because there are 27 Stage2
# concepts and representation structure requires >=2 samples/concept.
# ==============================================================================

MAX_SAMPLES=0


# ==============================================================================
# 8. Preflight
# ==============================================================================

echo "======================================================================"
echo "Geometry evaluation"
echo "======================================================================"
echo "Repository       : ${REPO_ROOT}"
echo "Evaluation code  : ${EVAL_DIR}"
echo
echo "Base model       : ${BASE_MODEL}"
echo "Stage1           : ${STAGE1_DIR}"
echo "Stage2           : ${STAGE2_DIR}"
echo "Stage2-only      : ${STAGE2_ONLY_DIR}"
echo
echo "Stage1 data      : ${STAGE1_DATA_DIR}"
echo "Stage2 data      : ${STAGE2_DATA_DIR}"
echo
echo "Stage1 log       : ${STAGE1_LOG}"
echo "Stage2 log       : ${STAGE2_LOG}"
echo "Stage2-only log  : ${STAGE2_ONLY_LOG}"
echo
echo "Result directory : ${RESULT_DIR}"
echo "Max samples      : ${MAX_SAMPLES}"
echo "======================================================================"
echo


mkdir -p "${RESULT_DIR}"


# ==============================================================================
# 9. Build command
# ==============================================================================

CMD=(
    python
    "${EVAL_DIR}/run_all.py"

    --eval-dir
    "${EVAL_DIR}"

    --stage1-data-dir
    "${STAGE1_DATA_DIR}"

    --stage2-data-dir
    "${STAGE2_DATA_DIR}"

    --base-model
    "${BASE_MODEL}"

    --stage1-dir
    "${STAGE1_DIR}"

    --stage2-dir
    "${STAGE2_DIR}"

    --stage2-only-dir
    "${STAGE2_ONLY_DIR}"

    --stage1-log
    "${STAGE1_LOG}"

    --stage2-log
    "${STAGE2_LOG}"

    --stage2-only-log
    "${STAGE2_ONLY_LOG}"

    --result-dir
    "${RESULT_DIR}"

    --max-samples
    "${MAX_SAMPLES}"
)


# Optional local WikiText corpus.
if [[ -n "${LANGUAGE_TEXT_FILE}" ]]; then
    CMD+=(
        --language-text-file
        "${LANGUAGE_TEXT_FILE}"
    )
fi


# ==============================================================================
# 10. Run
# ==============================================================================
#
# Do NOT use `set -e`.
#
# run_all.py itself continues through failed experiments and returns 1 only
# after all six evaluators have been attempted.
# ==============================================================================

"${CMD[@]}"
PIPELINE_STATUS=$?


# ==============================================================================
# 11. Final status
# ==============================================================================

echo
echo "======================================================================"

if [[ ${PIPELINE_STATUS} -eq 0 ]]; then
    echo "All evaluators completed successfully."
else
    echo "Evaluation finished with one or more failed evaluators."
    echo "The remaining evaluators were still attempted."
fi

echo
echo "Results:"
echo "  ${RESULT_DIR}"
echo
echo "Run summary:"
echo "  ${RESULT_DIR}/run_summary.md"
echo
echo "Download the whole result directory for analysis."
echo "======================================================================"

exit ${PIPELINE_STATUS}