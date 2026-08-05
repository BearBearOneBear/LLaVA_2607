@echo off
setlocal EnableExtensions
chcp 65001 >nul

REM ============================================================
REM Tiny Stage 1 and Stage 2 Training Test
REM
REM Anaconda Prompt에서 실행한다.
REM ============================================================

pushd "%~dp0..\.."
if errorlevel 1 (
    echo Failed to move to the LLaVA repository root.
    exit /b 1
)

set "REPOSITORY_ROOT=%CD%"
echo Repository root: %REPOSITORY_ROOT%


REM ------------------------------------------------------------
REM 실행 여부
REM ------------------------------------------------------------

if not defined RUN_DATA_CONVERSION set "RUN_DATA_CONVERSION=True"
if not defined RUN_MODEL_CREATION set "RUN_MODEL_CREATION=True"
if not defined RUN_TEST_DATA_CREATION set "RUN_TEST_DATA_CREATION=True"
if not defined RUN_STAGE1_TEST set "RUN_STAGE1_TEST=True"
if not defined RUN_STAGE2_TEST set "RUN_STAGE2_TEST=True"

if not defined RECREATE_TINY_MODEL set "RECREATE_TINY_MODEL=False"
if not defined RESET_OUTPUTS set "RESET_OUTPUTS=True"


REM ------------------------------------------------------------
REM 스크립트 경로
REM ------------------------------------------------------------

if not defined CONVERTER set "CONVERTER=tools\geometry\convert_stage1_parquet.py"
if not defined CREATE_MODEL_SCRIPT set "CREATE_MODEL_SCRIPT=tools\geometry\tiny_debug\create_tiny_model.py"
if not defined MAKE_TEST_SCRIPT set "MAKE_TEST_SCRIPT=tools\geometry\tiny_debug\make_test.py"
if not defined TRAIN_TINY_SCRIPT set "TRAIN_TINY_SCRIPT=tools\geometry\tiny_debug\train_tiny.py"


REM ------------------------------------------------------------
REM 원본 데이터 경로
REM ------------------------------------------------------------

if not defined STAGE1_DATA_DIR set "STAGE1_DATA_DIR=geometry_data\stage1_geometry_grounding"
if not defined STAGE2_DATA_DIR set "STAGE2_DATA_DIR=geometry_data\stage2_geometry_grounding"

if not defined STAGE1_TRAIN_PATH set "STAGE1_TRAIN_PATH=%STAGE1_DATA_DIR%\train.json"
if not defined STAGE1_IMAGE_FOLDER set "STAGE1_IMAGE_FOLDER=%STAGE1_DATA_DIR%\images"

if not defined STAGE2_TRAIN_PATH set "STAGE2_TRAIN_PATH=%STAGE2_DATA_DIR%\train.json"
if not defined STAGE2_EVAL_PATH set "STAGE2_EVAL_PATH=%STAGE2_DATA_DIR%\validation.json"
if not defined STAGE2_IMAGE_FOLDER set "STAGE2_IMAGE_FOLDER=%STAGE2_DATA_DIR%\images"


REM ------------------------------------------------------------
REM Tiny model 및 subset 경로
REM ------------------------------------------------------------

if not defined TINY_MODEL_DIR set "TINY_MODEL_DIR=debug_assets\tiny_llava"
if not defined TOKENIZER_PATH set "TOKENIZER_PATH=liuhaotian/llava-v1.5-7b"
if not defined VISION_TOWER set "VISION_TOWER=openai/clip-vit-large-patch14-336"

if not defined TEST_DATA_DIR set "TEST_DATA_DIR=debug_data"
if not defined NUM_TEST_SAMPLES set "NUM_TEST_SAMPLES=100"

if not defined STAGE1_TEST_TRAIN set "STAGE1_TEST_TRAIN=%TEST_DATA_DIR%\stage1\train_100.json"

if not defined STAGE2_TEST_TRAIN set "STAGE2_TEST_TRAIN=%TEST_DATA_DIR%\stage2\train_100.json"
if not defined STAGE2_TEST_EVAL set "STAGE2_TEST_EVAL=%TEST_DATA_DIR%\stage2\validation_100.json"

if not defined STAGE1_OUTPUT_DIR set "STAGE1_OUTPUT_DIR=debug_outputs\tiny_stage1"
if not defined STAGE2_OUTPUT_DIR set "STAGE2_OUTPUT_DIR=debug_outputs\tiny_stage2"

if not defined STAGE1_PROJECTOR_PATH set "STAGE1_PROJECTOR_PATH=%STAGE1_OUTPUT_DIR%\mm_projector.bin"


REM ------------------------------------------------------------
REM 학습 설정
REM ------------------------------------------------------------

if not defined MODEL_MAX_LENGTH set "MODEL_MAX_LENGTH=2048"
if not defined MAX_STEPS set "MAX_STEPS=20"

if not defined PER_DEVICE_TRAIN_BATCH_SIZE set "PER_DEVICE_TRAIN_BATCH_SIZE=1"
if not defined PER_DEVICE_EVAL_BATCH_SIZE set "PER_DEVICE_EVAL_BATCH_SIZE=1"
if not defined GRADIENT_ACCUMULATION_STEPS set "GRADIENT_ACCUMULATION_STEPS=1"

if not defined EVAL_STEPS set "EVAL_STEPS=10"
if not defined SAVE_STEPS set "SAVE_STEPS=10"
if not defined LOGGING_STEPS set "LOGGING_STEPS=1"

if not defined STAGE1_LEARNING_RATE set "STAGE1_LEARNING_RATE=1e-3"
if not defined STAGE2_LEARNING_RATE set "STAGE2_LEARNING_RATE=1e-4"

if not defined STAGE1_PROJECTOR_LR set "STAGE1_PROJECTOR_LR=1e-3"
if not defined STAGE2_PROJECTOR_LR set "STAGE2_PROJECTOR_LR=1e-4"


REM ------------------------------------------------------------
REM Python 환경
REM ------------------------------------------------------------

where python >nul 2>&1
if errorlevel 1 (
    echo Python was not found.
    echo Activate the Anaconda environment before running this file.
    goto :error
)

set "PYTHONPATH=%REPOSITORY_ROOT%;%PYTHONPATH%"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"

REM CPU tiny test
set "CUDA_VISIBLE_DEVICES="
set "TOKENIZERS_PARALLELISM=false"

python -c "import torch, transformers, pyarrow, tqdm, llava; print('Python environment is ready.')"
if errorlevel 1 goto :error


call :require_file "%CONVERTER%"
if errorlevel 1 goto :error

call :require_file "%CREATE_MODEL_SCRIPT%"
if errorlevel 1 goto :error

call :require_file "%MAKE_TEST_SCRIPT%"
if errorlevel 1 goto :error

call :require_file "%TRAIN_TINY_SCRIPT%"
if errorlevel 1 goto :error


call :require_directory "%STAGE1_DATA_DIR%\train_parquet"
if errorlevel 1 goto :error

call :require_directory "%STAGE2_DATA_DIR%\train_parquet"
if errorlevel 1 goto :error

call :require_directory "%STAGE2_DATA_DIR%\validation_parquet"
if errorlevel 1 goto :error


REM ============================================================
REM 1. Stage 1/2 데이터 변환
REM ============================================================

if /i "%RUN_DATA_CONVERSION%"=="True" (
    echo.
    echo Step 1: Converting Stage 1 train data.

    python "%CONVERTER%" ^
        --input_root "%STAGE1_DATA_DIR%" ^
        --output_dir "%STAGE1_DATA_DIR%" ^
        --splits train

    if errorlevel 1 goto :error

    echo.
    echo Step 1: Converting Stage 2 train and validation data.

    python "%CONVERTER%" ^
        --input_root "%STAGE2_DATA_DIR%" ^
        --output_dir "%STAGE2_DATA_DIR%" ^
        --splits train validation

    if errorlevel 1 goto :error
) else (
    echo Step 1: Dataset conversion skipped.
)


call :require_file "%STAGE1_TRAIN_PATH%"
if errorlevel 1 goto :error

call :require_directory "%STAGE1_IMAGE_FOLDER%\train"
if errorlevel 1 goto :error

call :require_file "%STAGE2_TRAIN_PATH%"
if errorlevel 1 goto :error

call :require_file "%STAGE2_EVAL_PATH%"
if errorlevel 1 goto :error

call :require_directory "%STAGE2_IMAGE_FOLDER%\train"
if errorlevel 1 goto :error

call :require_directory "%STAGE2_IMAGE_FOLDER%\validation"
if errorlevel 1 goto :error


REM ============================================================
REM 2. Tiny 모델 생성
REM ============================================================

if /i "%RUN_MODEL_CREATION%"=="True" (
    if exist "%TINY_MODEL_DIR%\config.json" if /i not "%RECREATE_TINY_MODEL%"=="True" (
        echo Step 2: Tiny model already exists. Creation skipped.
    ) else (
        call :create_tiny_model
        if errorlevel 1 goto :error
    )
) else (
    echo Step 2: Tiny model creation skipped.
)

call :require_file "%TINY_MODEL_DIR%\config.json"
if errorlevel 1 goto :error


REM ============================================================
REM 3. Tiny 데이터 생성
REM ============================================================

if /i "%RUN_TEST_DATA_CREATION%"=="True" (
    echo Step 3: Creating test datasets.

    python "%MAKE_TEST_SCRIPT%" ^
        --stage1_train_path "%STAGE1_TRAIN_PATH%" ^
        --stage2_train_path "%STAGE2_TRAIN_PATH%" ^
        --stage2_eval_path "%STAGE2_EVAL_PATH%" ^
        --output_dir "%TEST_DATA_DIR%" ^
        --num_samples "%NUM_TEST_SAMPLES%"

    if errorlevel 1 goto :error
) else (
    echo Step 3: Test dataset creation skipped.
)


call :require_file "%STAGE1_TEST_TRAIN%"
if errorlevel 1 goto :error

call :require_file "%STAGE2_TEST_TRAIN%"
if errorlevel 1 goto :error

call :require_file "%STAGE2_TEST_EVAL%"
if errorlevel 1 goto :error


if /i "%RESET_OUTPUTS%"=="True" (
    if exist "%STAGE1_OUTPUT_DIR%" rmdir /s /q "%STAGE1_OUTPUT_DIR%"
    if exist "%STAGE2_OUTPUT_DIR%" rmdir /s /q "%STAGE2_OUTPUT_DIR%"
)

if not exist "%STAGE1_OUTPUT_DIR%" mkdir "%STAGE1_OUTPUT_DIR%"
if not exist "%STAGE2_OUTPUT_DIR%" mkdir "%STAGE2_OUTPUT_DIR%"


REM ============================================================
REM 4. Tiny Stage 1 학습
REM
REM train-only, validation 없음
REM ============================================================

if /i "%RUN_STAGE1_TEST%"=="True" (
    echo Step 4: Starting tiny Stage 1 training.

    python "%TRAIN_TINY_SCRIPT%" ^
        --stage 1 ^
        --model_name_or_path "%TINY_MODEL_DIR%" ^
        --version "v1" ^
        --data_path "%STAGE1_TEST_TRAIN%" ^
        --image_folder "%STAGE1_IMAGE_FOLDER%" ^
        --vision_tower "%VISION_TOWER%" ^
        --mm_projector_type "mlp2x_gelu" ^
        --tune_mm_mlp_adapter True ^
        --mm_vision_select_layer -2 ^
        --mm_vision_select_feature "patch" ^
        --mm_use_im_start_end False ^
        --mm_use_im_patch_token False ^
        --image_aspect_ratio "pad" ^
        --output_dir "%STAGE1_OUTPUT_DIR%" ^
        --overwrite_output_dir True ^
        --num_train_epochs 1 ^
        --max_steps "%MAX_STEPS%" ^
        --per_device_train_batch_size "%PER_DEVICE_TRAIN_BATCH_SIZE%" ^
        --gradient_accumulation_steps "%GRADIENT_ACCUMULATION_STEPS%" ^
        --evaluation_strategy "no" ^
        --save_strategy "no" ^
        --learning_rate "%STAGE1_LEARNING_RATE%" ^
        --mm_projector_lr "%STAGE1_PROJECTOR_LR%" ^
        --weight_decay 0.0 ^
        --warmup_ratio 0.0 ^
        --lr_scheduler_type "cosine" ^
        --max_grad_norm 1.0 ^
        --optim "adamw_torch" ^
        --logging_steps "%LOGGING_STEPS%" ^
        --logging_first_step True ^
        --logging_nan_inf_filter False ^
        --bf16 False ^
        --fp16 False ^
        --tf32 False ^
        --bits 16 ^
        --model_max_length "%MODEL_MAX_LENGTH%" ^
        --gradient_checkpointing False ^
        --dataloader_num_workers 0 ^
        --dataloader_pin_memory False ^
        --lazy_preprocess True ^
        --report_to "none"

    if errorlevel 1 goto :error
) else (
    echo Step 4: Tiny Stage 1 training skipped.
)


if /i "%RUN_STAGE2_TEST%"=="True" (
    call :require_file "%STAGE1_PROJECTOR_PATH%"
    if errorlevel 1 goto :error
)


REM ============================================================
REM 5. Tiny Stage 2 학습
REM ============================================================

if /i "%RUN_STAGE2_TEST%"=="True" (
    echo Step 5: Starting tiny Stage 2 training.

    python "%TRAIN_TINY_SCRIPT%" ^
        --stage 2 ^
        --model_name_or_path "%TINY_MODEL_DIR%" ^
        --version "v1" ^
        --data_path "%STAGE2_TEST_TRAIN%" ^
        --eval_data_path "%STAGE2_TEST_EVAL%" ^
        --image_folder "%STAGE2_IMAGE_FOLDER%" ^
        --vision_tower "%VISION_TOWER%" ^
        --pretrain_mm_mlp_adapter "%STAGE1_PROJECTOR_PATH%" ^
        --mm_projector_type "mlp2x_gelu" ^
        --freeze_backbone False ^
        --tune_mm_mlp_adapter False ^
        --freeze_mm_mlp_adapter False ^
        --mm_vision_select_layer -2 ^
        --mm_vision_select_feature "patch" ^
        --mm_use_im_start_end False ^
        --mm_use_im_patch_token False ^
        --image_aspect_ratio "pad" ^
        --group_by_modality_length True ^
        --output_dir "%STAGE2_OUTPUT_DIR%" ^
        --overwrite_output_dir True ^
        --num_train_epochs 1 ^
        --max_steps "%MAX_STEPS%" ^
        --per_device_train_batch_size "%PER_DEVICE_TRAIN_BATCH_SIZE%" ^
        --per_device_eval_batch_size "%PER_DEVICE_EVAL_BATCH_SIZE%" ^
        --gradient_accumulation_steps "%GRADIENT_ACCUMULATION_STEPS%" ^
        --evaluation_strategy "steps" ^
        --eval_steps "%EVAL_STEPS%" ^
        --save_strategy "steps" ^
        --save_steps "%SAVE_STEPS%" ^
        --save_total_limit 1 ^
        --load_best_model_at_end False ^
        --learning_rate "%STAGE2_LEARNING_RATE%" ^
        --mm_projector_lr "%STAGE2_PROJECTOR_LR%" ^
        --weight_decay 0.0 ^
        --warmup_ratio 0.0 ^
        --lr_scheduler_type "cosine" ^
        --max_grad_norm 1.0 ^
        --optim "adamw_torch" ^
        --logging_steps "%LOGGING_STEPS%" ^
        --logging_first_step True ^
        --logging_nan_inf_filter False ^
        --bf16 False ^
        --fp16 False ^
        --tf32 False ^
        --bits 16 ^
        --model_max_length "%MODEL_MAX_LENGTH%" ^
        --gradient_checkpointing False ^
        --dataloader_num_workers 0 ^
        --dataloader_pin_memory False ^
        --lazy_preprocess True ^
        --report_to "none"

    if errorlevel 1 goto :error
) else (
    echo Step 5: Tiny Stage 2 training skipped.
)


echo.
echo Tiny Stage 1 and Stage 2 training test completed successfully.
echo Tiny model: %TINY_MODEL_DIR%
echo Stage 1 projector: %STAGE1_PROJECTOR_PATH%
echo Stage 1 output: %STAGE1_OUTPUT_DIR%
echo Stage 2 output: %STAGE2_OUTPUT_DIR%

popd
endlocal
exit /b 0


:create_tiny_model
echo Step 2: Creating the tiny LLaVA model.

if exist "%TINY_MODEL_DIR%" (
    python "%CREATE_MODEL_SCRIPT%" ^
        --tokenizer_name_or_path "%TOKENIZER_PATH%" ^
        --output_dir "%TINY_MODEL_DIR%" ^
        --overwrite
) else (
    python "%CREATE_MODEL_SCRIPT%" ^
        --tokenizer_name_or_path "%TOKENIZER_PATH%" ^
        --output_dir "%TINY_MODEL_DIR%"
)

if errorlevel 1 exit /b 1
exit /b 0


:require_file
if not exist "%~1" (
    echo Required file was not found: %~1
    exit /b 1
)
exit /b 0


:require_directory
if not exist "%~1\" (
    echo Required directory was not found: %~1
    exit /b 1
)
exit /b 0


:error
echo.
echo Tiny training test stopped because an error occurred.

popd
endlocal
exit /b 1