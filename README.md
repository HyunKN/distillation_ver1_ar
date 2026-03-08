# MobileCLIP2 Retrieval Optimization

> LPCVC 2026 Track 1 — MobileCLIP2 기반 이미지-텍스트 검색 경량 모델 학습 및 지식 증류

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## Overview

Apple `MobileCLIP2-S0`를 학생 모델로, 두 개의 대형 teacher 모델로 지식 증류를 수행해 모바일 환경 이미지-텍스트 검색 모델을 학습합니다.

**핵심 사항**
- **학생 모델**: `MobileCLIP2-S0` — 사전학습된 멀티모달 경량 모델 파인튜닝
- **Teacher 모델**: `ViT-gopt-16-SigLIP2-256` + `PE-Core-bigG-14-448`
- **Dual Distillation**: 배치/샘플 품질 기반 adaptive teacher weighting
- **Offline Feature 모드**: teacher VRAM 없이 반복 실험 가능
- **배포 경로**: ONNX export → Qualcomm AI Hub compile/profile

---

## 문서 안내

> 자세한 프로젝트 방향과 코드 상세는 아래 문서를 참고하세요.

| 문서 | 역할 |
|------|------|
| `docs/PROJECT_MAP.md` | 프로젝트 전체 구조, 방향, 운영 기본값, 학습 로드맵 — 길잡이 |
| `docs/PROJECT_GUIDE.md` | 코드 구조, 함수 상세, 학습/증류/배포 흐름 — 세부 매뉴얼 |
| `docs/README.md` | 문서 인덱스 및 읽는 순서 |
| `docs/archive/` | 완료된 handover 및 정리 기록 |

---

## Architecture

```text
┌────────────────────────────────────────────────────────────┐
│                        Teacher Models                      │
│  ┌────────────────────────┐   ┌─────────────────────────┐  │
│  │ ViT-gopt-16-SigLIP2-256│   │  PE-Core-bigG-14-448   │  │
│  │     (text-image align) │   │ (visual robustness)    │  │
│  └────────────┬───────────┘   └────────────┬────────────┘  │
│               │                            │               │
│               └──── adaptive teacher weighting ───────────┘│
│                                │                           │
└────────────────────────────────┼───────────────────────────┘
                                 ▼
                  ┌─────────────────────────────────┐
                  │     MobileCLIP2-S0 Student      │
                  │  image encoder + text encoder   │
                  │   contrastive + distill losses  │
                  └─────────────────────────────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
            Retrieval Evaluation          ONNX Export
           (I2T / T2I Recall@K)        + QAI Hub Compile/Profile
```

### 모델 및 구조 설명

#### 1. 학생 모델 — MobileCLIP2-S0

실제 배포 대상입니다. 이미지와 텍스트를 각각 임베딩으로 변환하고, 두 임베딩의 유사도로 검색을 수행합니다.

- 코드: `src/lpcvc_retrieval/mobileclip2.py`
- 팩토리: `src/lpcvc_retrieval/model.py`
- 기본 임베딩 차원: `256`

#### 2. Teacher 모델 — SigLIP 2 + PE-Core

두 teacher 모두 `open_clip`으로 로드합니다.

- `ViT-gopt-16-SigLIP2-256` — 이미지-텍스트 정렬 품질이 강한 teacher
- `PE-Core-bigG-14-448` — 시각적 강건성이 강한 teacher
- `distill.teachers` — 어떤 teacher를 쓸지 정의하는 목록
- `distill.static_teacher_weights` — `teacher_weight_mode=static`일 때만 쓰는 mixing 설정

#### 3. 현재 기본 Distillation 방식

```yaml
distill:
  adaptive_teacher_weight: true
  static_teacher_weights: [0.5, 0.5]
  teacher_weight_mode: adaptive
  source_teacher_weights: {}
```

- teacher 둘 다 사용
- source prior 없이 순수 adaptive routing
- 현재 배치/샘플에서 더 우수한 teacher에 비중을 부여
- `static_teacher_weights`는 현재 기본 모드(`adaptive`)에서는 최종 mixing 비율에 영향을 주지 않음

#### 4. Offline Teacher Feature 모드

teacher를 매 스텝 실행하지 않고, 미리 teacher 임베딩을 `.pt` 파일로 추출해 학습에 재사용할 수 있습니다.

- VRAM 절감
- 반복 실험 속도 개선
- teacher 출력 재현성 확보

---

## Quick Start

### 1. Installation

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

의존성 설치:

```bash
pip install -r requirements.txt
pip install -e .
pip install git+https://github.com/apple/ml-mobileclip.git
```

> **참고** — `run_train.py`는 내부에서 `src/`를 추가하므로 바로 실행 가능합니다.
> `scripts/*.py`는 `pip install -e .`를 해두면 `PYTHONPATH` 설정 부담이 줄어듭니다.

### 2. Dataset Setup

현재 기본 데이터 모드는 JSONL입니다.

```yaml
data:
  mode: jsonl
  image_root: dataset
  train_jsonl: dataset/prepared_jsonl/train.jsonl
  val_jsonl: dataset/prepared_jsonl/val.jsonl
```

JSONL 한 줄 형식:

```json
{"image":"coco/train2017/000000000009.jpg","captions":["a cat on sofa"],"source":"coco"}
```

지원하는 `source` 키: `coco`, `flickr30k`, `open_images`, `wit`

**전처리**

Windows:

```bash
python scripts\preprocess\parse_lpcvc_sources.py --data_root D:\LPCVC_Data --out_dir D:\LPCVC_Data\prepared_jsonl --val_ratio 0.01
```

Linux/macOS:

```bash
python scripts/preprocess/parse_lpcvc_sources.py --data_root /data/LPCVC_Data --out_dir /data/LPCVC_Data/prepared_jsonl --val_ratio 0.01
```

학습 전에 `config.yaml` 또는 `--override`로 경로를 맞춥니다.

```bash
python run_train.py --config config.yaml \
  --override data.image_root=D:/LPCVC_Data \
  --override data.train_jsonl=D:/LPCVC_Data/prepared_jsonl/train.jsonl \
  --override data.val_jsonl=D:/LPCVC_Data/prepared_jsonl/val.jsonl
```

### 2.5 Offline Teacher Feature Extraction (권장)

Windows:

```bash
python scripts\extract_features.py --config config.yaml --out_dir features --override data.train_augment=false
```

Linux/macOS:

```bash
python scripts/extract_features.py --config config.yaml --out_dir features --override data.train_augment=false
```

생성되는 파일:

| 파일 | 설명 |
|------|------|
| `features/teacher_0_train.pt` | Teacher 0 학습셋 임베딩 |
| `features/teacher_1_train.pt` | Teacher 1 학습셋 임베딩 |
| `features/teacher_0_val.pt` | Teacher 0 검증셋 임베딩 |
| `features/teacher_1_val.pt` | Teacher 1 검증셋 임베딩 |

각 파일에 저장되는 메타: `img_embs`, `txt_embs`, `caption_indices`, `sample_count`, `dataset_fingerprint`

학습 시 `OfflineFeatureDataset`가 데이터셋 길이, 임베딩 shape, `caption_indices`, `dataset_fingerprint` 일치를 검증합니다.

사용 설정:

```yaml
distill:
  offline_feature_dir: features
```

### 3. Training

기본 학습:

```bash
python run_train.py --config config.yaml
```

자주 쓰는 예시:

```bash
# 빠른 테스트
python run_train.py --config config.yaml --override train.epochs=1 --override data.batch_size=32

# teacher 없이 학습
python run_train.py --config config.yaml --override distill.use_teacher=false

# offline feature 모드
python run_train.py --config config.yaml --override distill.offline_feature_dir=features
```

기본 학습 스택:

| 항목 | 설정 |
|------|------|
| Optimizer | AdamW |
| Scheduler | warmup + cosine decay |
| AMP | CUDA에서 기본 활성 |
| EMA | 기본 활성 |
| `torch.compile` | 기본 활성 |
| Best checkpoint 기준 | I2T R@10 |

### 4. Evaluation

Windows:

```bash
set PYTHONPATH=src && python scripts\eval.py --config config.yaml --ckpt runs\lpcvc_clip_lite\best.pt
```

Linux/macOS:

```bash
PYTHONPATH=src python scripts/eval.py --config config.yaml --ckpt runs/lpcvc_clip_lite/best.pt
```

- `image_id`가 있으면 COCO-style 양방향 평가
- 없으면 index 기반 legacy retrieval 평가

### 5. Export to ONNX

Windows:

```bash
set PYTHONPATH=src && python scripts\export_onnx_split.py --config config.yaml --ckpt runs\lpcvc_clip_lite\best.pt --out_dir exported_onnx
```

Linux/macOS:

```bash
PYTHONPATH=src python scripts/export_onnx_split.py --config config.yaml --ckpt runs/lpcvc_clip_lite/best.pt --out_dir exported_onnx
```

출력: `exported_onnx/image_encoder.onnx`, `exported_onnx/text_encoder.onnx`

### 6. Qualcomm AI Hub Compile / Profile

최초 1회 인증:

```bash
qai-hub configure --api_token <YOUR_QAI_HUB_TOKEN>
```

컴파일 + 프로파일:

```bash
python compile_and_profile.py \
  --onnx_dir exported_onnx \
  --img_name image_encoder.onnx \
  --txt_name text_encoder.onnx \
  --device "XR2 Gen 2 (Proxy)"
```

컴파일만 수행 (`--skip_profile`):

```bash
python compile_and_profile.py \
  --onnx_dir exported_onnx \
  --img_name image_encoder.onnx \
  --txt_name text_encoder.onnx \
  --device "XR2 Gen 2 (Proxy)" \
  --skip_profile
```

입력 스펙: 이미지 `(1, 3, 224, 224)` float32 / 텍스트 `(1, 77)` int32

---

## Project Structure

```text
.
├── README.md                        # 실행 가이드 (이 파일)
├── LICENSE                          # MIT License
├── THIRD_PARTY_LICENSES.md          # 외부 모델/데이터 라이선스 정보
├── config.yaml                      # 학습 설정
├── pyproject.toml                   # 패키지 메타데이터
├── requirements.txt                 # 의존성
├── run_train.py                     # 학습 엔트리포인트
├── compile_and_profile.py           # QAI Hub 컴파일/프로파일
│
├── docs/
│   ├── README.md                    # 문서 인덱스
│   ├── PROJECT_GUIDE.md             # 코드 구조 및 기술 상세
│   └── archive/                     # 완료된 기록
│       ├── ROOT_DOCS_CLEANUP_SUMMARY.md
│       └── completed-tasks/
│           └── 260228_COMPARISON_AND_TRAINING_METHODS_HANDOVER.md
│
├── scripts/
│   ├── eval.py                      # 체크포인트 평가
│   ├── export_onnx_split.py         # ONNX 분리 export
│   ├── extract_features.py          # Offline teacher feature 추출
│   └── preprocess/
│       ├── parse_lpcvc_sources.py   # JSONL 전처리
│       └── materialize_upload_subset.py
│
└── src/lpcvc_retrieval/
    ├── __init__.py
    ├── config.py                    # 설정 로딩/파싱
    ├── data.py                      # 데이터 파이프라인
    ├── distill.py                   # Teacher 로딩 및 distillation
    ├── ema.py                       # Exponential Moving Average
    ├── export.py                    # ONNX export 유틸
    ├── logger.py                    # 학습 로거
    ├── losses.py                    # 손실 함수 모음
    ├── metrics.py                   # Retrieval 평가 지표
    ├── mobileclip2.py               # MobileCLIP2 학생 모델 래퍼
    ├── model.py                     # 모델 팩토리
    └── train.py                     # 학습 루프
```

---

## Configuration

자주 확인하는 설정:

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `data.mode` | `jsonl` | 데이터 로더 모드 |
| `data.batch_size` | `128` | 학습 배치 크기 |
| `data.max_captions_per_image` | `5` | 캡션 샘플링 상한 |
| `model.mobileclip2_variant` | `S0` | 학생 모델 variant |
| `model.embed_dim` | `256` | 최종 임베딩 차원 |
| `distill.use_teacher` | `true` | Teacher distillation 사용 |
| `distill.adaptive_teacher_weight` | `true` | 동적 가중치 활성화 |
| `distill.teacher_weight_mode` | `adaptive` | Teacher routing 모드 |
| `distill.static_teacher_weights` | `[0.5, 0.5]` | `static` 모드 전용 mixing 비율 |
| `distill.source_teacher_weights` | `{}` | `adaptive_source` 전용 source prior |
| `distill.offline_feature_dir` | `null` | Offline feature 경로 |
| `loss.w_distill_affinity` | `0.8` | Affinity distillation 비중 |
| `train.use_compile` | `true` | `torch.compile` 사용 |
| `train.use_ema` | `true` | EMA 사용 |
| `output.out_dir` | `runs/lpcvc_clip_lite` | 체크포인트 출력 경로 |

---

## Performance Targets

| 항목 | 현재 코드 기준 |
|------|----------------|
| 주 평가 지표 | I2T / T2I Recall@1, 5, 10 |
| Best checkpoint 선정 | I2T R@10 |
| 학생 입력 해상도 | 224 x 224 |
| 텍스트 길이 | 77 |
| Teacher 입력 크기 | SigLIP2 256 / PE-Core 448 |
| Distillation 방식 | contrastive + affinity distill |
| 배포 산출물 | `image_encoder.onnx`, `text_encoder.onnx` |
| 디바이스 검증 | Qualcomm AI Hub compile/profile |

> 이 저장소는 측정 스크립트와 export/compile 경로를 제공합니다.
> 실제 latency/메모리 수치는 export된 모델과 디바이스 job 결과로 별도 확인해야 합니다.

---

## Technical Details

상세한 기술 설명은 [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)를 참조하세요.

주요 내용:
- 데이터 파이프라인과 JSONL 계약
- MobileCLIP2 학생 모델 래퍼 구조
- Teacher 로딩과 adaptive distillation 로직
- Offline teacher feature 검증 방식
- 학습 / 평가 / export / QAI Hub 흐름

---

## References

- [MobileCLIP](https://github.com/apple/ml-mobileclip) — Apple ML Research
- [SigLIP 2](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md) — Google Research
- [PE-Core / Perception Models](https://github.com/facebookresearch/perception_models) — Meta FAIR
- [OpenCLIP](https://github.com/mlfoundations/open_clip) — MLFoundations
- [TinyCLIP](https://arxiv.org/abs/2309.12314)

---

## License

프로젝트 코드는 MIT License를 따릅니다.

- 저장소 라이선스: [LICENSE](LICENSE)
- 외부 구성요소: [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)

> 모델 가중치와 데이터셋은 별도 라이선스를 가질 수 있습니다.
> 재배포 또는 상업 사용 전 upstream 모델 카드와 데이터셋 약관을 반드시 재확인하세요.

---

## Acknowledgements

- Apple ML Research
- Google Research / DeepMind
- Meta FAIR
- OpenCLIP contributors
- LPCVC 2026 organizers
