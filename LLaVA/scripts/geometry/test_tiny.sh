#!/bin/bash

set -euo pipefail


# ============================================================
# Tiny Stage 1 and Stage 2 Training Test
#
# 1. 3-layer tiny LLaVA 모델 생성
# 2. 작은 테스트 데이터셋 생성
# 3. Stage 1 projector train-only 학습
# 4. Stage 1 projector 저장 확인
# 5. Stage 2 LLM + projector 학습
#
# DeepSpeed를 사용하지 않고 CPU에서 실행한다.
# LLaVA 저장소 루트에서 실행한다.
# ============================================================


# ------------------------------------------------------------
# 실행 여부
# ------------------------------------------------------------

RUN_MODEL_CREATION="${RUN_MODEL_CREATION:-True}"
RUN_DATA_CREATION="${RUN_DATA_CREATION:-True}"
RUN_STAGE1_TEST="${RUN_STAGE1_TEST:-True}"
RUN_STAGE2_TEST="${RUN_STAGE2_TEST:-True}"

RECREATE_TINY_MODEL="${RECREATE_TINY_MODEL:-False}"
RESET_OUTPUTS="${RESET_OUTPUTS:-True}"


# ------------------------------------------------------------
# 스크립트 경로
# ------------------------------------------------------------

CREATE_MODEL_SCRIPT="${CREATE_MODEL_SCRIPT:-./tools/geometry/tiny_debug/create_tiny_model.py}"
MAKE_TEST_SCRIPT="${MAKE_TEST_SCRIPT:-./tools/geometry/tiny_debug/make_test.py}"
TRAIN_TINY_SCRIPT="${TRAIN_TINY_SCRIPT:-./tools/geometry/tiny_debug/train_tiny.py}"


# ------------------------------------------------------------
# 원본 데이터 경로
# ------------------------------------------------------------

STAGE1_DATA_DIR="${STAGE1_DATA_DIR:-./geometry_data/stage1_geometry_grounding}"
STAGE2_DATA_DIR="${STAGE2_DATA_DIR:-./geometry_data/stage2_geometry_grounding}"

STAGE1_TRAIN_PATH="${STAGE1_TRAIN_PATH:-${STAGE1_DATA_DIR}/train.json}"
STAGE1_IMAGE_FOLDER="${STAGE1_IMAGE_FOLDER:-${STAGE1_DATA_DIR}/images}"

STAGE2_TRAIN_PATH="${STAGE2_TRAIN_PATH:-${STAGE2_DATA_DIR}/train.json}"
STAGE2_EVAL_PATH="${STAGE2_EVAL_PATH:-${STAGE2_DATA_DIR}/validation.json}"
STAGE2_IMAGE_FOLDER="${STAGE2_IMAGE_FOLDER:-${STAGE2_DATA_DIR}/images}"


# ------------------------------------------------------------
# Tiny 모델
# ------------------------------------------------------------

TINY_MODEL_DIR="${TINY_MODEL_DIR:-./debug_assets/tiny_llava}"

TOKENIZER_PATH="${TOKENIZER_PATH:-liuhaotian/llava-v1.5-7b}"
VISION_TOWER="${VISION_TOWER:-openai/clip-vit-large-patch14-336}"


# ------------------------------------------------------------
# Tiny 데이터 경로
# ------------------------------------------------------------

TEST_DATA_DIR="${TEST_DATA_DIR:-./debug_data}"
NUM_TEST_SAMPLES="${NUM_TEST_SAMPLES:-100}"

STAGE1_TEST_TRAIN="${STAGE1_TEST_TRAIN:-${TEST_DATA_DIR}/stage1/train_100.json}"

STAGE2_TEST_TRAIN="${STAGE2_TEST_TRAIN:-${TEST_DATA_DIR}/stage2/train_100.json}"
STAGE2_TEST_EVAL="${STAGE2_TEST_EVAL:-${TEST_DATA_DIR}/stage2/validation_100.json}"


# ------------------------------------------------------------
# 출력 경로
# ------------------------------------------------------------

STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-./debug_outputs/tiny_stage1}"
STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-./debug_outputs/tiny_stage2}"

STAGE1_PROJECTOR_PATH="${STAGE1_PROJECTOR_PATH:-${STAGE1_OUTPUT_DIR}/mm_projector.bin}"


# ------------------------------------------------------------
# 로그 경로
# ------------------------------------------------------------

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-./logs/tiny_test/${RUN_ID}}"

mkdir -p "${LOG_DIR}"


# ------------------------------------------------------------
# 학습 설정
# ------------------------------------------------------------

MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"

MAX_STEPS="${MAX_STEPS:-20}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"

EVAL_STEPS="${EVAL_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-10}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"

STAGE1_LEARNING_RATE="${STAGE1_LEARNING_RATE:-1e-3}"
STAGE2_LEARNING_RATE="${STAGE2_LEARNING_RATE:-1e-4}"

STAGE1_PROJECTOR_LR="${STAGE1_PROJECTOR_LR:-1e-3}"
STAGE2_PROJECTOR_LR="${STAGE2_PROJECTOR_LR:-1e-4}"


is_true() {
    [[ "${1,,}" == "true" ]]
}


require_file() {
    local FILE_PATH="$1"

    if [[ ! -f "${FILE_PATH}" ]]; then
        echo "Required file was not found: ${FILE_PATH}" >&2
        exit 1
    fi
}


require_directory() {
    local DIRECTORY_PATH="$1"

    if [[ ! -d "${DIRECTORY_PATH}" ]]; then
        echo "Required directory was not found: ${DIRECTORY_PATH}" >&2
        exit 1
    fi
}


# ------------------------------------------------------------
# 필수 스크립트와 데이터 확인
# ------------------------------------------------------------

require_file "${CREATE_MODEL_SCRIPT}"
require_file "${MAKE_TEST_SCRIPT}"
require_file "${TRAIN_TINY_SCRIPT}"

require_file "${STAGE1_TRAIN_PATH}"
require_directory "${STAGE1_IMAGE_FOLDER}"

require_file "${STAGE2_TRAIN_PATH}"
require_file "${STAGE2_EVAL_PATH}"
require_directory "${STAGE2_IMAGE_FOLDER}"


echo "Starting tiny Stage 1 and Stage 2 training test."
echo "Tiny model directory: ${TINY_MODEL_DIR}"
echo "Vision tower: ${VISION_TOWER}"
echo "Test data directory: ${TEST_DATA_DIR}"
echo "Stage 1 output directory: ${STAGE1_OUTPUT_DIR}"
echo "Stage 2 output directory: ${STAGE2_OUTPUT_DIR}"
echo "Maximum steps: ${MAX_STEPS}"
echo "Log directory: ${LOG_DIR}"


# ============================================================
# 1. Tiny model 생성
# ============================================================

if is_true "${RUN_MODEL_CREATION}"; then
    if [[ -f "${TINY_MODEL_DIR}/config.json" ]] \
        && ! is_true "${RECREATE_TINY_MODEL}"; then

        echo "Step 1: Tiny model already exists."
        echo "Step 1: Model creation skipped."
    else
        echo "Step 1: Creating the tiny LLaVA model."

        MODEL_ARGUMENTS=(
            --tokenizer_name_or_path "${TOKENIZER_PATH}"
            --output_dir "${TINY_MODEL_DIR}"
        )

        if [[ -d "${TINY_MODEL_DIR}" ]]; then
            MODEL_ARGUMENTS+=(--overwrite)
        fi

        python "${CREATE_MODEL_SCRIPT}" \
            "${MODEL_ARGUMENTS[@]}" \
            2>&1 | tee "${LOG_DIR}/01_create_tiny_model.log"

        echo "Step 1: Tiny model creation completed."
    fi
else
    echo "Step 1: Tiny model creation skipped."
fi

require_file "${TINY_MODEL_DIR}/config.json"


# ============================================================
# 2. Tiny dataset 생성
# ============================================================

if is_true "${RUN_DATA_CREATION}"; then
    echo "Step 2: Creating test datasets."

    python "${MAKE_TEST_SCRIPT}" \
        --stage1_train_path "${STAGE1_TRAIN_PATH}" \
        --stage2_train_path "${STAGE2_TRAIN_PATH}" \
        --stage2_eval_path "${STAGE2_EVAL_PATH}" \
        --output_dir "${TEST_DATA_DIR}" \
        --num_samples "${NUM_TEST_SAMPLES}" \
        2>&1 | tee "${LOG_DIR}/02_make_test_data.log"

    echo "Step 2: Test dataset creation completed."
else
    echo "Step 2: Test dataset creation skipped."
fi

require_file "${STAGE1_TEST_TRAIN}"
require_file "${STAGE2_TEST_TRAIN}"
require_file "${STAGE2_TEST_EVAL}"


# ------------------------------------------------------------
# 기존 출력 초기화
# ------------------------------------------------------------

if is_true "${RESET_OUTPUTS}"; then
    rm -rf "${STAGE1_OUTPUT_DIR}"
    rm -rf "${STAGE2_OUTPUT_DIR}"
fi

mkdir -p "${STAGE1_OUTPUT_DIR}"
mkdir -p "${STAGE2_OUTPUT_DIR}"


# CPU만 사용한다.
export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM="false"


# ============================================================
# 3. Tiny Stage 1 학습
#
# Stage 1은 train-only, validation 없음
# ============================================================

if is_true "${RUN_STAGE1_TEST}"; then
    echo "Step 3: Starting tiny Stage 1 training."

    python "${TRAIN_TINY_SCRIPT}" \
        --stage 1 \
        --model_name_or_path "${TINY_MODEL_DIR}" \
        --version "v1" \
        --data_path "${STAGE1_TEST_TRAIN}" \
        --image_folder "${STAGE1_IMAGE_FOLDER}" \
        --vision_tower "${VISION_TOWER}" \
        --mm_projector_type "mlp2x_gelu" \
        --tune_mm_mlp_adapter True \
        --mm_vision_select_layer -2 \
        --mm_vision_select_feature "patch" \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio "pad" \
        --output_dir "${STAGE1_OUTPUT_DIR}" \
        --overwrite_output_dir True \
        --num_train_epochs 1 \
        --max_steps "${MAX_STEPS}" \
        --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --evaluation_strategy "no" \
        --save_strategy "no" \
        --learning_rate "${STAGE1_LEARNING_RATE}" \
        --mm_projector_lr "${STAGE1_PROJECTOR_LR}" \
        --weight_decay 0.0 \
        --warmup_ratio 0.0 \
        --lr_scheduler_type "cosine" \
        --max_grad_norm 1.0 \
        --optim "adamw_torch" \
        --logging_steps "${LOGGING_STEPS}" \
        --logging_first_step True \
        --logging_nan_inf_filter False \
        --bf16 False \
        --fp16 False \
        --tf32 False \
        --bits 16 \
        --model_max_length "${MODEL_MAX_LENGTH}" \
        --gradient_checkpointing False \
        --dataloader_num_workers 0 \
        --dataloader_pin_memory False \
        --lazy_preprocess True \
        --report_to "none" \
        2>&1 | tee "${LOG_DIR}/03_stage1_training.log"

    echo "Step 3: Tiny Stage 1 training completed."
else
    echo "Step 3: Tiny Stage 1 training skipped."
fi


# Stage 2를 실행하려면 Stage 1 projector가 필요하다.
if is_true "${RUN_STAGE2_TEST}"; then
    require_file "${STAGE1_PROJECTOR_PATH}"
fi


# ============================================================
# 4. Tiny Stage 2 학습
# ============================================================

if is_true "${RUN_STAGE2_TEST}"; then
    echo "Step 4: Starting tiny Stage 2 training."

    python "${TRAIN_TINY_SCRIPT}" \
        --stage 2 \
        --model_name_or_path "${TINY_MODEL_DIR}" \
        --version "v1" \
        --data_path "${STAGE2_TEST_TRAIN}" \
        --eval_data_path "${STAGE2_TEST_EVAL}" \
        --image_folder "${STAGE2_IMAGE_FOLDER}" \
        --vision_tower "${VISION_TOWER}" \
        --pretrain_mm_mlp_adapter "${STAGE1_PROJECTOR_PATH}" \
        --mm_projector_type "mlp2x_gelu" \
        --freeze_backbone False \
        --tune_mm_mlp_adapter False \
        --freeze_mm_mlp_adapter False \
        --mm_vision_select_layer -2 \
        --mm_vision_select_feature "patch" \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio "pad" \
        --group_by_modality_length True \
        --output_dir "${STAGE2_OUTPUT_DIR}" \
        --overwrite_output_dir True \
        --num_train_epochs 1 \
        --max_steps "${MAX_STEPS}" \
        --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
        --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --evaluation_strategy "steps" \
        --eval_steps "${EVAL_STEPS}" \
        --save_strategy "steps" \
        --save_steps "${SAVE_STEPS}" \
        --save_total_limit 1 \
        --load_best_model_at_end False \
        --learning_rate "${STAGE2_LEARNING_RATE}" \
        --mm_projector_lr "${STAGE2_PROJECTOR_LR}" \
        --weight_decay 0.0 \
        --warmup_ratio 0.0 \
        --lr_scheduler_type "cosine" \
        --max_grad_norm 1.0 \
        --optim "adamw_torch" \
        --logging_steps "${LOGGING_STEPS}" \
        --logging_first_step True \
        --logging_nan_inf_filter False \
        --bf16 False \
        --fp16 False \
        --tf32 False \
        --bits 16 \
        --model_max_length "${MODEL_MAX_LENGTH}" \
        --gradient_checkpointing False \
        --dataloader_num_workers 0 \
        --dataloader_pin_memory False \
        --lazy_preprocess True \
        --report_to "none" \
        2>&1 | tee "${LOG_DIR}/04_stage2_training.log"

    echo "Step 4: Tiny Stage 2 training completed."
else
    echo "Step 4: Tiny Stage 2 training skipped."
fi


echo "Tiny Stage 1 and Stage 2 training test completed successfully."
echo "Tiny model: ${TINY_MODEL_DIR}"
echo "Stage 1 projector: ${STAGE1_PROJECTOR_PATH}"
echo "Stage 1 output: ${STAGE1_OUTPUT_DIR}"
echo "Stage 2 output: ${STAGE2_OUTPUT_DIR}"
echo "Logs: ${LOG_DIR}"