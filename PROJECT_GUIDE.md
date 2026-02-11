# MobileCLIP2 Retrieval Optimization 프로젝트 가이드

이 문서는 두 가지를 동시에 목표로 합니다.

1. 아이나 비전공자도 큰 흐름을 이해할 수 있게 설명
2. 실제로 실행/학습/배포할 수 있을 만큼 자세하게 설명

문서 기준: 2026-02-11, 현재 저장소 코드 기준

---

## 1. 이 프로젝트를 아주 쉽게 설명하면?

### 아이에게 설명하는 버전

이 프로젝트는 "사진 찾기 AI"를 만드는 작업입니다.

- 사람이 "소파 위 고양이 사진 찾아줘"라고 말하면
- AI가 사진들을 훑어보고
- 가장 비슷한 사진을 위에 보여줍니다.

즉, 말과 사진을 같은 "의미 지도" 위에 올려서 가까운 것을 찾는 기술입니다.

### 비전공자에게 설명하는 버전

이미지와 텍스트를 각각 숫자 벡터(embedding)로 바꾸고, 두 벡터의 유사도를 계산해서 검색합니다.

- 이미지 인코더: 사진 -> 벡터
- 텍스트 인코더: 문장 -> 벡터
- 검색: 벡터끼리 코사인 유사도 비교

이 방식은 모바일에서도 빠르게 동작할 수 있어서 대회 요구사항(경량/저지연)에 맞추기 좋습니다.

---

## 2. 이 프로젝트가 실제로 하는 일 (끝까지 한 줄 흐름)

1. 데이터셋(COCO 또는 JSONL)에서 이미지-캡션 쌍을 읽음
2. 학생 모델(MobileCLIP2)이 이미지/텍스트 임베딩을 만듦
3. 필요하면 큰 Teacher 모델들이 만든 임베딩과 관계(affinity)를 따라 배우게 함
4. 학습 후 `best.pt` 저장
5. 학습된 모델을 `image_encoder.onnx` / `text_encoder.onnx`로 분리 export
6. Qualcomm AI Hub에서 컴파일/프로파일하여 디바이스 성능 확인

---

## 3. 학생/선생님 모델 구조를 쉽게 이해하기

### 학생 모델 (실제 배포용)

- 모델: `MobileCLIP2` (기본 `S4`)
- 역할: 실제로 디바이스에서 실행되는 가벼운 모델
- 코드: `src/lpcvc_retrieval/mobileclip2.py`

### 선생님 모델 (학습 도우미)

기본 `config.yaml` 기준:

- Teacher 1: `ViT-gopt-16-SigLIP2-256` + `webli` (가중치 0.6)
- Teacher 2: `ViT-H-14-worldwide-quickgelu` + `metaclip2_worldwide` (가중치 0.4, OpenCLIP 체크포인트 약 7.44GB)

중요:

- 현재 기본은 MetaCLIP 2 Worldwide(H/14) 조합입니다.
- 어떤 모델을 받는지는 Hugging Face 링크 문자열 자체보다 `open_clip`의 pretrained registry 해석이 기준입니다.

---

## 4. "학습"을 처음 보는 사람을 위한 설명

학습은 "정답 맞히기 + 선생님 따라하기"를 반복하는 과정입니다.

1. 한 배치의 이미지/문장을 가져옴
2. 학생 모델이 각각 임베딩을 생성
3. 대조학습 손실로 "짝이 맞는 이미지-문장은 가까워지고, 틀린 짝은 멀어지게" 학습
4. 선생님이 켜져 있으면, 선생님의 임베딩 관계를 학생이 모방하도록 distillation 손실 추가
5. 역전파(backprop)로 가중치 업데이트
6. 이 과정을 여러 epoch 반복

현재 코드에서 쓰는 주요 loss:

- `siglip_loss` (기본 대조학습)
- `pairwise_ranking_loss` (옵션)
- `hard_negative_contrastive_loss` (옵션)
- `text_text_contrastive_loss` (옵션)
- `compute_affinity_distill_loss` (teacher 사용 시)

추가 안정화:

- AMP (mixed precision)
- cosine LR + warmup
- gradient clipping
- EMA
- `torch.compile` (옵션)

---

## 5. 폴더/파일 역할 (처음 보는 사람 기준)

```text
.
├── run_train.py
├── config.yaml
├── compile_and_profile.py
├── scripts/
│   ├── eval.py
│   └── export_onnx_split.py
└── src/lpcvc_retrieval/
    ├── config.py
    ├── data.py
    ├── distill.py
    ├── ema.py
    ├── export.py
    ├── logger.py
    ├── losses.py
    ├── metrics.py
    ├── mobileclip2.py
    ├── model.py
    └── train.py
```

핵심 파일 한 줄 요약:

- `run_train.py`: 학습 시작 버튼
- `config.yaml`: 모든 옵션을 모아둔 설정 파일
- `src/lpcvc_retrieval/train.py`: 진짜 학습 루프
- `src/lpcvc_retrieval/data.py`: 데이터 읽기/전처리
- `src/lpcvc_retrieval/distill.py`: teacher 로딩과 distillation 계산
- `scripts/eval.py`: 학습된 모델 성능 계산
- `scripts/export_onnx_split.py`: 이미지/텍스트 인코더를 ONNX로 분리 저장
- `compile_and_profile.py`: QAI Hub에 컴파일/프로파일 작업 제출

---

## 6. `config.yaml`를 자세히 이해하기

현재 코드가 실제로 읽는 루트 키:

- `data`, `device`, `distill`, `export`, `loss`, `model`, `output`, `seed`, `train`

자주 만지는 키와 의미:

- `data.mode`: `coco` 또는 `jsonl`
- `data.max_captions_per_image`: 한 이미지에서 학습 시 샘플링할 캡션 수 상한
- `model.mobileclip2_variant`: 학생 모델 크기 (`S0`~`S4` 등)
- `model.embed_dim`: 최종 임베딩 차원
- `distill.use_teacher`: teacher distillation 사용 여부
- `loss.w_distill_affinity`: distillation 비중
- `loss.w_hard_negative`, `loss.w_text_text`: 추가 loss 비중
- `train.lr`, `train.epochs`, `train.warmup_epochs`, `train.grad_clip`
- `train.use_compile`, `train.use_ema`, `train.ema_decay`
- `output.out_dir`: 체크포인트 저장 폴더 (기본 `runs/lpcvc_clip_lite`)

### 초심자용 튜닝 가이드

- VRAM이 부족하면: `data.batch_size` 먼저 낮추기
- 학습이 불안정하면: `train.lr` 낮추기, `train.warmup_epochs` 늘리기
- teacher가 너무 무거우면: `distill.use_teacher: false`로 스모크 테스트
- 속도 테스트용은: `train.epochs: 1`로 먼저 파이프라인 검증

---

## 7. 실행 방법 (Windows CMD 기준으로 상세)

### 7.1 가상환경

```bat
python -m venv venv
venv\Scripts\activate
```

프롬프트 앞에 `(venv)`가 보이면 활성화된 상태입니다.

### 7.2 패키지 설치

```bat
pip install -r requirements.txt
pip install git+https://github.com/apple/ml-mobileclip.git
```

ONNX exporter 에러(`onnxscript`)가 나면:

```bat
pip install onnxscript
```

### 7.3 학습

```bat
python run_train.py --config config.yaml
```

학습 완료 후 기본 출력:

- `runs\lpcvc_clip_lite\best.pt`
- `runs\lpcvc_clip_lite\last.pt`
- `runs\lpcvc_clip_lite\epoch_*.pt`

### 7.4 평가

`scripts`는 `PYTHONPATH=src`가 필요합니다.

```bat
set PYTHONPATH=src && python scripts\eval.py --config config.yaml --ckpt runs\lpcvc_clip_lite\best.pt
```

### 7.5 ONNX 분리 export

```bat
set PYTHONPATH=src && python scripts\export_onnx_split.py --config config.yaml --ckpt runs\lpcvc_clip_lite\best.pt --out_dir exported_onnx
```

출력:

- `exported_onnx\image_encoder.onnx`
- `exported_onnx\text_encoder.onnx`
- 필요 시 `*.onnx.data`

### 7.6 Qualcomm AI Hub 컴파일/프로파일

최초 1회 토큰 설정:

```bat
qai-hub configure --api_token <YOUR_QAI_HUB_TOKEN>
```

컴파일 + 프로파일:

```bat
python compile_and_profile.py --onnx_dir exported_onnx --img_name image_encoder.onnx --txt_name text_encoder.onnx --device "XR2 Gen 2 (Proxy)"
```

컴파일만 하고 싶으면:

```bat
python compile_and_profile.py --onnx_dir exported_onnx --img_name image_encoder.onnx --txt_name text_encoder.onnx --device "XR2 Gen 2 (Proxy)" --skip_profile
```

주의:

- `--skip_profile`면 PROFILE 탭에 안 뜨고 COMPILE 탭만 뜹니다.
- 이는 정상 동작입니다.

---

## 8. `.onnx.data` 파일이 왜 생기나요?

쉽게 말하면:

- `.onnx`는 설계도
- `.onnx.data`는 큰 부품(가중치) 창고

모델이 크면 ONNX가 자동으로 외부 데이터 포맷을 씁니다.  
그래서 두 파일이 세트로 생길 수 있고, 이상이 아닙니다.

중요한 규칙:

- `.onnx`와 `.onnx.data`를 같은 폴더에 함께 둬야 함
- 업로드/컴파일 시 둘을 같이 참조해야 정상 동작

---

## 9. 데이터셋 모드 상세

### COCO 모드 (`data.mode: coco`)

필수 구조:

```text
dataset/coco/
├── train2017/
├── val2017/
└── annotations/
    ├── captions_train2017.json
    └── captions_val2017.json
```

특징:

- 학습에서는 한 이미지의 여러 캡션 중 랜덤 샘플
- 평가에서는 캡션 단위로 펼쳐서 정확하게 recall 계산
- `image_id`를 유지해서 COCO 스타일 평가를 수행

### JSONL 모드 (`data.mode: jsonl`)

예시 한 줄:

```json
{"image":"train2017/000000000009.jpg","captions":["a cat on sofa","indoor cat photo"]}
```

---

## 10. 자주 막히는 문제와 해결

1. `ModuleNotFoundError: lpcvc_retrieval`

- 원인: `scripts/*.py` 실행 시 `PYTHONPATH=src` 누락
- 해결: `set PYTHONPATH=src && ...` 형태로 실행

2. `No module named 'onnxscript'`

- 원인: PyTorch ONNX exporter 의존성 누락
- 해결: `pip install onnxscript`

3. PROFILE 탭에 새 작업이 안 보임

- 원인: `--skip_profile`로 컴파일만 실행
- 해결: `--skip_profile` 없이 다시 실행하거나 compile job 기준으로 profile만 추가 제출

4. `deactivate` 명령이 안 됨

- 원인: 이미 venv 비활성 상태일 수 있음
- 확인: 프롬프트에 `(venv)`가 없으면 비활성 상태

5. Teacher 모델 다운로드가 느리거나 실패

- 원인: 네트워크/HF rate limit
- 해결: 토큰 설정, 재시도, 먼저 스모크 설정으로 파이프라인 점검

---

## 11. 프로젝트를 수정할 때 꼭 같이 업데이트할 문서

아래 셋은 항상 같이 맞춰야 혼선이 없습니다.

- `config.yaml` (실제 실행 기준)
- `README.md` (빠른 시작)
- `PROJECT_GUIDE.md` (상세 설명)

특히 모델 이름/체크포인트 경로/AI Hub 옵션을 바꾼 경우, 세 문서를 동시에 갱신하세요.

---

## 12. 마지막 정리 (정말 처음 보는 사람용)

이 프로젝트는 "사진과 문장을 같은 언어로 바꾸는 번역기"를 만드는 일입니다.

- 학생 모델은 작고 빠른 번역기
- 선생님 모델은 정확한 번역기
- 학습은 학생이 선생님을 따라 하며 실력을 키우는 과정
- 완성 후 ONNX로 내보내고, 실제 디바이스 환경(QAI Hub)에서 속도/메모리를 검증

이 흐름만 기억하면, 나머지 설정과 코드 구조가 훨씬 쉽게 읽힙니다.

---

## 13. 코드 전체 해설 (파일별/함수별)

이 섹션은 "저장소의 코드 파일을 하나씩" 설명합니다.
처음 코드를 읽는 사람이 "어디부터 봐야 하는지"와 "각 함수가 왜 있는지"를 파악하는 용도입니다.

### 13.1 전체 호출 흐름 (큰 그림)

학습 흐름:

`run_train.py` -> `src/lpcvc_retrieval/config.py::load_config` -> `src/lpcvc_retrieval/train.py::train` -> 내부에서 `data.py`, `model.py`, `distill.py`, `losses.py`, `metrics.py`, `ema.py`, `logger.py` 사용

평가 흐름:

`scripts/eval.py` -> `config.py`, `data.py`, `model.py`, `metrics.py`

배포 흐름:

`scripts/export_onnx_split.py` -> `model.py` -> `export.py::export_onnx_split`

QAI Hub 흐름:

`compile_and_profile.py` -> ONNX 검증 -> compile job -> (옵션) profile job

### 13.2 엔트리 파일들

### `run_train.py`

목적:

- 사용자 명령(`python run_train.py --config ...`)을 받아 학습 파이프라인 시작

핵심 함수:

- `main()`
  - argparse로 `--config` 수신
  - `load_config` 호출
  - GPU 정보 출력
  - `train(cfg)` 실행
  - 최종 best checkpoint 경로 출력

왜 중요한가:

- `src` path를 내부에서 추가하므로, 학습 실행 시 `PYTHONPATH`를 별도로 주지 않아도 됨

### `scripts/eval.py`

목적:

- 저장된 체크포인트를 로드해 retrieval recall 계산

핵심 함수:

- `main()`
  - `--config`, `--ckpt`, `--override` 지원
  - 모델/데이터 로드 후 임베딩 전체 계산
  - `image_id`가 있으면 COCO-style 평가(`coco_bidirectional_recall`)
  - 없으면 index 기반 legacy 평가(`bidirectional_recall`)

주의:

- `scripts` 실행이므로 `PYTHONPATH=src` 필요

### `scripts/export_onnx_split.py`

목적:

- 체크포인트를 로드해 이미지 인코더/텍스트 인코더를 ONNX 두 파일로 export

핵심 함수:

- `main()`
  - `--config`, `--ckpt`, `--out_dir`, `--prefix`, `--add_timestamp`, `--override` 지원
  - `export.py::export_onnx_split` 호출

### `compile_and_profile.py`

목적:

- ONNX를 QAI Hub에 올려 compile/profile 자동화

핵심 함수:

- `compile_model(...)`: compile job 제출 및 완료 대기
- `run_profile(...)`: profile job 제출
- `main()`
  - ONNX 파일 존재/유효성 검사
  - `--device` 기준 compile 실행
  - `--skip_profile`가 없으면 profile도 실행

핵심 옵션:

- 런타임: `qnn_dlc`
- `--skip_profile`: compile만 수행 (PROFILE 탭 작업 생성 안 됨)

### 13.3 패키지 파일별 상세 (`src/lpcvc_retrieval`)

### `config.py`

역할:

- YAML 설정을 읽고 dot-access 가능한 객체로 제공

핵심 구성:

- `class CfgNode`: `cfg.train.lr` 같은 접근 지원
- `_deep_set(...)`: `a.b.c=value` 형태 override 적용
- `_parse_value(...)`: 문자열 override를 타입(bool/number/list)으로 변환
- `load_config(...)`: YAML + override 로드
- `resolve_device(...)`: `auto/cuda/cpu`를 실제 실행 디바이스 문자열로 변환

### `data.py`

역할:

- 토크나이저 생성, 데이터셋 로딩, 이미지 전처리, 배치 콜레이트

핵심 함수/클래스:

- `build_tokenizer()`
  - `openai/clip-vit-base-patch32` 토크나이저 사용
- `_img_transform_train(...)`, `_img_transform_eval(...)`
  - 학습/평가용 전처리 분리
- `class JsonlRetrievalDataset`
  - JSONL 형식 로더
  - 학습 시 캡션 랜덤 샘플링
  - 평가 시 캡션 단위 샘플 생성
- `class CocoCaptionsRetrievalDataset`
  - COCO captions json 직접 로딩
  - `image_id`/`ann_id` 메타 유지
- `make_datasets(cfg, tokenizer)`
  - `data.mode`에 따라 COCO/JSONL 분기 생성
- `collate_fn(batch)`
  - `(imgs, toks, metas)` 형태로 배치 구성

왜 중요한가:

- 이 프로젝트에서 COCO-style 정답 매칭의 핵심인 `image_id`를 데이터 단계에서 유지함

### `mobileclip2.py`

역할:

- MobileCLIP2 학생 모델 래퍼

핵심 구성:

- `class MobileCLIP2Student`
  - `VARIANT_MAP`으로 모델 이름 매핑
  - `_load_model()`
    - `open_clip.create_model_and_transforms`로 모델 로드
    - 기본 pretrained는 `dfndr2b` (체크포인트 경로 미지정 시)
    - 가능하면 `ml-mobileclip`의 reparameterization 적용
    - 출력 차원 불일치 시 `image_proj/text_proj` 선형층 추가
  - `encode_image(...)`, `encode_text(...)`
    - 각 인코더 결과를 L2 normalize
  - `forward(images, text_input)`
    - 이미지/텍스트 임베딩 동시 반환
- `create_mobileclip2_model(...)`
  - 팩토리 함수

### `model.py`

역할:

- 설정 기반 모델 생성과 ONNX export wrapper 제공

핵심 구성:

- `create_model_from_config(cfg, ...)`
  - `MobileCLIP2Student` 생성
- `class OnnxWrapper`
  - `(image, text_input)` -> `(img_emb, txt_emb)` 출력 형태로 묶음

### `train.py`

역할:

- 실제 학습/검증의 중심 파일

핵심 함수:

- `set_seed(seed)`: 재현성 기본 세팅
- `get_cosine_schedule_with_warmup(...)`: warmup + cosine LR 스케줄
- `evaluate(...)`
  - 검증 데이터 임베딩 생성
  - 가능하면 COCO-style 양방향 recall 계산
- `_normalize_clip(...)`
  - CLIP normalize 유틸 (현재 메인 흐름에서 직접 사용되지는 않음)
- `train(cfg)`
  - 모델/데이터/optimizer/scheduler/scaler 초기화
  - teacher 초기화(`distill.use_teacher`)
  - 배치 루프에서 losses 계산 및 역전파
  - EMA 적용 평가
  - `best.pt`, `last.pt`, `epoch_*.pt` 저장

loss 조합 위치:

- `w_contrastive` (siglip)
- `w_rank`
- `w_hard_negative`
- `w_text_text`
- `w_distill_affinity`

### `distill.py`

역할:

- Teacher 모델 로딩과 distillation loss 계산

핵심 구성:

- `TeacherConfig`, `DistillConfig` 데이터클래스
- `class OpenClipTeacher`
  - teacher별 입력 크기 조정, 정규화(mean/std), 자체 tokenizer 사용
- `class EnsembleTeacher`
  - teacher 여러 개를 묶어 forward
- `create_teacher(cfg, device)`
  - single/ensemble 분기 생성
- `compute_affinity_distill_loss(...)`
  - student similarity와 teacher similarity의 KL 정렬
  - teacher weight 반영
  - selective masking 옵션 반영

중요 구현 포인트:

- mean/std 버퍼를 teacher와 같은 device에 올려 device mismatch를 방지

### `losses.py`

역할:

- 학습에서 쓰는 손실 함수 모음

함수 설명:

- `clip_contrastive_loss(...)`
  - 전통 CLIP 대칭 cross-entropy
- `multi_gt_masked_contrastive_loss(...)`
  - 같은 `image_id`를 모두 positive로 처리하는 multi-GT 버전
- `siglip_loss(...)`
  - sigmoid 기반 pairwise 손실 (현재 기본 대조 손실)
- `pairwise_ranking_loss(...)`
  - hard negative top-k margin ranking
- `hard_negative_contrastive_loss(...)`
  - 가장 혼동되는 negative에 집중
- `text_text_contrastive_loss(...)`
  - 같은 이미지의 다른 캡션끼리 가깝게 학습

### `metrics.py`

역할:

- retrieval 평가 지표 계산

함수 설명:

- `recall_at_k(...)`: 일반 Recall@K
- `recall_at_1_5_10(...)`: legacy helper
- `bidirectional_recall(...)`: I2T/T2I 양방향 계산
- `format_metrics(...)`: 로그 문자열 생성
- `coco_i2t_recall(...)`, `coco_t2i_recall(...)`
  - `image_id` 매칭 기반 COCO 정식 스타일 계산
- `coco_bidirectional_recall(...)`
  - 위 두 결과를 통합

### `export.py`

역할:

- PyTorch 모델을 ONNX로 내보내는 로직

함수 설명:

- `export_onnx(...)`
  - 단일 wrapper 모델 형태 export (레거시 용도)
- `export_onnx_split(...)`
  - `image_encoder.onnx` / `text_encoder.onnx` 분리 export
  - text 입력 dtype은 `int32`

### `ema.py`

역할:

- EMA(Exponential Moving Average) 가중치 관리

핵심 메서드:

- `_init_shadow()`: shadow 파라미터 초기화
- `update()`: 학습 스텝마다 EMA 갱신
- `apply_shadow()`: 평가 시 EMA 가중치 적용
- `restore()`: 원래 가중치 복원
- `state_dict()`, `load_state_dict()`: 체크포인트 저장/로드

### `logger.py`

역할:

- WandB를 선택적으로 붙이기 위한 얇은 래퍼

핵심 메서드:

- `log(...)`: step metric 기록
- `log_epoch(...)`: epoch metric 기록
- `finish()`: run 종료

특징:

- `use_wandb=false`면 WandB 없이도 코드가 깨지지 않도록 설계됨

### `__init__.py`

역할:

- `lpcvc_retrieval`를 파이썬 패키지로 인식시키는 기준 파일

### 13.4 코드 읽기 추천 순서 (완전 초심자용)

아래 순서로 보면 이해가 가장 빠릅니다.

1. `run_train.py`
2. `src/lpcvc_retrieval/train.py`
3. `src/lpcvc_retrieval/data.py`
4. `src/lpcvc_retrieval/mobileclip2.py`
5. `src/lpcvc_retrieval/distill.py`
6. `src/lpcvc_retrieval/losses.py`
7. `src/lpcvc_retrieval/metrics.py`
8. `scripts/eval.py`
9. `scripts/export_onnx_split.py`
10. `compile_and_profile.py`

---

## 14. 논문/모델 출처 및 라이선스(저작권) 정리

이 섹션은 업로드/공유 시 필요한 출처 표기를 한 번에 확인하도록 만든 체크리스트입니다.

### 14.1 현재 설정 기준 실제 사용 모델 (검증 포함)

`config.yaml`의 teacher 설정이 실제로 어떤 체크포인트를 로드하는지는 `open_clip.pretrained.get_pretrained_cfg`로 확인했습니다.

- Teacher 1 설정: `ViT-gopt-16-SigLIP2-256` + `webli`
  - 실제 hf_hub: `timm/ViT-gopt-16-SigLIP2-256`
- Teacher 2 설정: `ViT-H-14-worldwide-quickgelu` + `metaclip2_worldwide`
  - 실제 hf_hub: `timm/vit_huge_patch14_clip_224.metaclip2_worldwide`
  - 참고 용량: `open_clip_model.safetensors` 약 7.44GB, `model.safetensors` 약 2.53GB (모델 카드 파일 목록 기준)
  - 파일 목록 링크: https://huggingface.co/timm/vit_huge_patch14_clip_224.metaclip2_worldwide/tree/main , https://huggingface.co/facebook/metaclip-2-worldwide-huge-quickgelu/tree/main

### 14.2 모델 링크 + 라이선스/저작권

| 구분 | 실제 사용/출처 | GitHub | Hugging Face | 라이선스/저작권 메모 |
|---|---|---|---|---|
| Student 모델 | Apple MobileCLIP2-S4 | https://github.com/apple/ml-mobileclip | https://huggingface.co/apple/MobileCLIP2-S4 | HF 표기: `apple-amlr`. Apple 저장소에는 `Code: MIT`, `ML models: Apple ML Research Model TOU`로 명시됨 |
| Teacher 1 | SigLIP2 Giant 계열 (`ViT-gopt-16-SigLIP2-256`) | https://github.com/google-research/big_vision | https://huggingface.co/timm/ViT-gopt-16-SigLIP2-256 (또는 https://huggingface.co/google/siglip2-giant-opt-patch16-256) | HF 표기: `apache-2.0` |
| Teacher 2 | MetaCLIP 2 Worldwide H/14 (`ViT-H-14-worldwide-quickgelu`) | https://github.com/facebookresearch/MetaCLIP | https://huggingface.co/timm/vit_huge_patch14_clip_224.metaclip2_worldwide (OpenCLIP/timm), https://huggingface.co/facebook/metaclip-2-worldwide-huge-quickgelu (facebook), https://huggingface.co/collections/facebook/meta-clip-2 | HF 모델 카드 표기 기준 `cc-by-nc-4.0` (비상업 조항 포함, 사용 전 재확인 권장) |
| 로딩 프레임워크 | OpenCLIP | https://github.com/mlfoundations/open_clip | (모델 허브는 각 모델 카드 참조) | OpenCLIP 저장소 라이선스: MIT |

권장 표기(실무):

- 모델 재배포/상업 사용 전, 각 모델 카드의 라이선스와 원문 LICENSE 파일을 반드시 재확인
- 특히 `cc-by-nc-4.0`은 상업적 이용이 제한될 수 있으므로 대회/프로덕션 용도 구분 필요
- `apple-amlr`/`Apple ML Research Model TOU`도 사용 범위 조건을 먼저 확인해야 함

### 14.3 논문 출처(실제 링크)

아래는 이 저장소의 모델/손실/증류 설명에 직접 연결되는 주요 논문입니다.

| 주제 | 논문 | 공식 링크 |
|---|---|---|
| 기본 VLM 대조학습 | CLIP: Learning Transferable Visual Models From Natural Language Supervision | https://arxiv.org/abs/2103.00020 |
| Sigmoid 기반 CLIP 학습 | SigLIP: Sigmoid Loss for Language Image Pre-Training | https://arxiv.org/abs/2303.15343 |
| Teacher 계열(최신) | SigLIP 2: Multilingual Vision-Language Encoders... | https://arxiv.org/abs/2502.14786 |
| Student 기반 모델 | MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training | https://arxiv.org/abs/2311.17049 |
| Student 최신 버전 | MobileCLIP2: Improving Multi-Modal Reinforced Training | https://openreview.net/forum?id=WeF9zolng8 |
| MetaCLIP 계열 | Demystifying CLIP Data | https://arxiv.org/abs/2309.16671 |
| MetaCLIP 2 계열 | Meta CLIP 2: A Worldwide Scaling Recipe | https://arxiv.org/abs/2507.22062 |
| Affinity distillation 계열 | TinyCLIP: CLIP Distillation via Affinity Mimicking and Weight Inheritance | https://arxiv.org/abs/2309.12314 |
| Hard negative 아이디어 참고 | BLIP: Bootstrapping Language-Image Pre-training... | https://arxiv.org/abs/2201.12086 |

참고:

- 코드 주석의 `FG-CLIP`, `TULIP` 표기는 "아이디어 출처" 수준으로 사용되고 있으며, 현재 저장소 문맥에서 특정 단일 버전 논문을 1:1로 고정 인용하지는 않았습니다.
- 필요 시 팀 내부 기준으로 `FG-CLIP`/`TULIP`의 정확한 목표 논문 버전을 확정해 이 표에 추가하세요.
- 논문 본문/그림의 저작권은 각 논문의 저자 및 출판 정책(arXiv/OpenReview/학회)에 따르며, 이 저장소는 논문을 인용해 구현 아이디어를 사용합니다.

### 14.4 저작권 표기 문구 예시 (README/Handover용)

아래 문구를 저장소 문서에 그대로 사용해도 됩니다.

```text
This project uses third-party models and implementations:
- MobileCLIP2 (Apple): Code MIT, model weights under Apple ML Research Model TOU (see model card/license).
- SigLIP2 checkpoints: Apache-2.0 (see Hugging Face model card).
- MetaCLIP 2 Worldwide H/14 checkpoint used via OpenCLIP/timm: CC-BY-NC-4.0.
- OpenCLIP framework: MIT.

Please verify each upstream license for your intended use (especially commercial usage).
```
