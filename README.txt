안녕하세요. 늦어서 죄송합니다.

---------------------------------------------------------------------------
LLaVA 1.5 기반 Stage 1-2 학습 코드를 정리하여 GitHub에 업로드했습니다.

https://github.com/BearBearOneBear/LLaVA_2607

LLaVA 1.5 코드를 기반으로 하였습니다.
일부 tool과 script를 작성하였습니다.
geometry_data에 stage 1, 2, 3에 해당하는 데이터가 parquet 형식으로 들어있습니다.

---------------------------------------------------------------------------
현재 파이프라인은 아래 과정을 하나의 스크립트로 순차 실행하도록 구성했습니다.
1. Stage 1, 2 Parquet 데이터 변환
2. Stage 1 smoke test
3. Stage 1 mlp 학습
4. Stage 1 최적 mlp 선택
5. Stage 2 smoke test
6. Stage 2 LLM + mlp 학습

중간 단계에서 오류가 발생하면 중단되며, 단계별 로그가 별도 저장됩니다.

---------------------------------------------------------------------------
이하는 gpt로 작성한 환경 설정입니다.

저장소를 clone한 뒤 내부 LLaVA 폴더로 이동합니다.
Python 3.10 환경을 만들고 LLaVA 학습 패키지를 설치합니다.
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install pyarrow tqdm

pip install flash-attn --no-build-isolation (FlashAttention 2 지원 환경에서)

---------------------------------------------------------------------------
GPU 환경에 따라 아래 두 파일의 설정을 조정해야 합니다.
scripts/geometry/train_stage1.sh
scripts/geometry/train_stage2.sh

GPU 모델
GPU 개수
GPU당 VRAM
CUDA 버전
CPU RAM
checkpoint 저장 가능 용량
사용 가능한 GPU 번호

GPU 개수에 따른 global batch 조정이 필요합니다.
GPU 수 × GPU당 train batch × gradient accumulation steps
현재 global batch는 32입니다.

---------------------------------------------------------------------------
다음 모델을 사용합니다.
liuhaotian/llava-v1.5-7b
openai/clip-vit-large-patch14-336
서버에 모델 cache가 없다면 최초 실행 시 Hugging Face에서 다운로드합니다.

---------------------------------------------------------------------------
설정이 끝나면 다음 스크립트를 실행하면 진행되길 희망합니다..
bash scripts/geometry/run_stage1_stage2_pipeline.sh

---------------------------------------------------------------------------