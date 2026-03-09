# Dual Distillation Retrieval Optimization

> LPCVC 2026 Track 1 — MobileNetV4 Hybrid Large + DatologyAI retr-opt-vit-b-32 기반 이미지-텍스트 검색 경량 모델 학습 및 지식 증류

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## Overview

현재 브랜치에서는 `MobileNetV4 Hybrid Large` 이미지 타워와 `hf-hub:DatologyAI/retr-opt-vit-b-32` 텍스트 타워를 결합한 dual-tower 학생 모델을 사용해 모바일 환경 이미지-텍스트 검색 모델을 학습합니다.

**핵심 사항**
- **학생 모델**: `DualTowerStudent` — `MobileNetV4 Hybrid Large` image encoder + `DatologyAI/retr-opt-vit-b-32` text encoder
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
| `docs/archive/` | 필요 시 완료 기록을 보관하는 아카이브 영역 |

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
                  ┌──────────────────────────────────────────┐
                  │        Dual-Tower Student (current)      │
                  │  Image: MobileNetV4 Hybrid Large         │
                  │  Text : DatologyAI/retr-opt-vit-b-32     │
                  │  contrastive + distill losses            │
                  └──────────────────────────────────────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
            Retrieval Evaluation          ONNX Export
           (I2T / T2I Recall@K)        + QAI Hub Compile/Profile
```

### 모델 및 구조 설명

#### 1. 학생 모델 — Dual-Tower Student

실제 배포 대상입니다. 이미지 타워는 `MobileNetV4 Hybrid Large`, 텍스트 타워는 `hf-hub:DatologyAI/retr-opt-vit-b-32`를 사용하며, 두 타워의 임베딩을 같은 `256`차원으로 투영한 뒤 유사도로 검색을 수행합니다. 학생 텍스트 타워는 retrieval-optimized OpenCLIP HF Hub 경로로 직접 로드되며, LPCVC 텍스트 입력 규격인 `1 x 77`과 맞습니다.

기본 설정에서는 `freeze_image_backbone=false`, `freeze_text_backbone=false`이므로 이미지/텍스트 인코더 둘 다 학습됩니다. 여기서 `freeze=true`는 해당 타워를 고정해 업데이트하지 않는다는 뜻입니다.

- 코드: `src/lpcvc_retrieval/dual_tower.py`
- 팩토리: `src/lpcvc_retrieval/model.py`
- 기본 이미지 입력 크기: `384 x 384`
- 텍스트 길이: `77`
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

입력 스펙: 이미지 `(1, 3, 384, 384)` float32 / 텍스트 `(1, 77)` int32

---

## Validation Snapshot

2026-03-10 기준 현재 브랜치에서 아래 경로를 실제로 다시 확인했습니다.

| 항목 | 결과 | 비고 |
|------|------|------|
| Small teacher feature extraction | 통과 | `ViT-B-32 (openai)` + `ViT-B-16 (openai)` 경로는 그대로 동작 |
| Student tokenizer/model load | 통과 | `hf-hub:DatologyAI/retr-opt-vit-b-32`, `context_length=77` |
| Student dummy forward | 통과 | `(1,3,384,384)` + `(1,77)` -> image/text `(1,256)` |
| Online teacher train smoke | 통과 | small teacher 2개(`ViT-B-32`, `ViT-B-16`), 1 epoch CPU smoke |
| Offline feature extraction smoke | 통과 | `tmp/smoke_e2e/features` |
| Offline teacher train smoke | 통과 | `tmp/smoke_e2e/runs_offline` |
| Eval smoke | 통과 | online/offline best checkpoint 모두 `scripts/eval.py`로 재검증 |
| ONNX split export smoke | 통과 | `tmp/smoke_e2e/exported_onnx_local/{image_encoder,text_encoder}.onnx` |
| AI Hub upload-only smoke | 통과 | image=`mn493d1rq`, text=`mmx26z7kn` |

> AI Hub compile/profile는 이번 확인 범위에 포함하지 않았습니다. 이번 smoke는 ONNX export와 upload까지만 확인했습니다.

---

## Project Structure

```text
.
├── README.md                        # 실행 가이드 (이 파일)
├── LICENSE                          # Apache-2.0 License
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
│   └── archive/                     # 필요 시 완료 기록을 보관하는 아카이브 영역
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
    ├── dual_tower.py                # 현재 학생 모델 구현 (MobileNetV4 + DatologyAI retr-opt)
    ├── model.py                     # 모델 팩토리
    └── train.py                     # 학습 루프
```

---

## Configuration

자주 확인하는 설정만 요약합니다. 전체 키별 동작은 [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)의 Configuration Reference를 참고하세요.

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `data.mode` | `jsonl` | `jsonl`이면 JSONL 데이터셋, `coco`면 COCO captions JSON 로더를 사용 |
| `data.batch_size` | `128` | train/eval/extract에 공통으로 쓰이는 기본 배치 크기 |
| `data.max_captions_per_image` | `5` | 학습 시 한 이미지에서 캡션 후보를 몇 개까지 샘플링할지 결정 |
| `data.train_augment` | `true` | `true`면 train transform에 랜덤 crop/flip/color jitter 적용 |
| `data.tokenizer_type` | `open_clip` | 현재 student text tower와 맞는 OpenCLIP tokenizer 경로 사용 |
| `model.image_model_name` | `mobilenetv4_hybrid_large.e600_r384_in1k` | student 이미지 타워 이름 (`timm.create_model`) |
| `model.text_model_name` | `hf-hub:DatologyAI/retr-opt-vit-b-32` | student 텍스트 타워 이름 (`open_clip.create_model`) |
| `model.freeze_image_backbone` | `false` | `false`면 이미지 타워 학습, `true`면 고정 |
| `model.freeze_text_backbone` | `false` | `false`면 텍스트 타워 학습, `true`면 고정 |
| `model.image_input_size` | `384` | student 이미지 입력 리사이즈 크기 및 export 입력 크기 |
| `model.embed_dim` | `256` | 최종 검색 임베딩 차원 |
| `distill.use_teacher` | `true` | `true`면 teacher distillation loss를 함께 학습 |
| `distill.teacher_weight_mode` | `adaptive` | teacher mixing 방식을 `static`/`adaptive`/`adaptive_source` 중 선택 |
| `distill.offline_feature_dir` | `null` | 경로가 있으면 online teacher 대신 pre-extracted teacher feature 사용 |
| `loss.w_contrastive` | `1.0` | 기본 SigLIP retrieval loss 비중 |
| `loss.w_distill_affinity` | `0.8` | teacher affinity distillation 비중 |
| `train.epochs` | `10` | 총 학습 epoch 수 |
| `train.lr` | `5e-4` | AdamW 기본 learning rate |
| `train.amp` | `true` | CUDA에서 mixed precision 학습 사용 |
| `train.use_compile` | `true` | 가능하면 `torch.compile`로 모델을 컴파일 |
| `train.use_ema` | `true` | EMA shadow weights를 유지하고 평가 시 사용 |
| `output.out_dir` | `runs/lpcvc_clip_lite` | 체크포인트와 학습 산출물 저장 경로 |

---

## Performance Targets

| 항목 | 현재 코드 기준 |
|------|----------------|
| 주 평가 지표 | I2T / T2I Recall@1, 5, 10 |
| Best checkpoint 선정 | I2T R@10 |
| 학생 입력 해상도 | 384 x 384 |
| 텍스트 길이 | 77 |
| Teacher 입력 크기 | SigLIP2 256 / PE-Core 448 |
| Distillation 방식 | contrastive + affinity distill |
| 배포 산출물 | `image_encoder.onnx`, `text_encoder.onnx` |
| 디바이스 검증 | Qualcomm AI Hub compile/profile |

> 이 저장소는 측정 스크립트와 export/compile 경로를 제공합니다.
> 실제 latency/메모리 수치는 export된 모델과 디바이스 job 결과로 별도 확인해야 합니다.

### Student Encoder Runtime Snapshot

측정 기준: `XR2 Gen 2 (Proxy)`에서 student image/text encoder를 각각 독립적으로 compile/profile한 결과입니다.

| Encoder | Model | Input | Minimum Inference Time | Estimated Peak Memory Usage | Compute Units | Compile Job ID | Profile Job ID |
|------|------|------|------------------------|-----------------------------|---------------|----------------|----------------|
| Image Encoder | `mobilenetv4_hybrid_large.e600_r384_in1k` | `384x384` | `15.85 ms` | `2 - 116 MB` | `NPU 354` | `-` | `j57dw6er5` |
| Text Encoder | `DatologyAI/retr-opt-vit-b-32` | `int32[1,77]` | `4.1 ms` | `0 - 146 MB` | `NPU 478` | `jpryqw30g` | `jpydwm38p` |

- image encoder는 Qualcomm AI Hub에 image encoder만 직접 업로드해 profile한 결과입니다.
- text encoder는 `DatologyAI/retr-opt-vit-b-32` profile 결과 기준입니다.
- 두 값을 단순 합산한 최소 추론시간은 약 `19.95 ms`지만, 실제 end-to-end 파이프라인 latency와 동일하다고 보기는 어렵습니다.

---

## Technical Details

상세한 기술 설명은 [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)를 참조하세요.

주요 내용:
- 데이터 파이프라인과 JSONL 계약
- Dual-tower 학생 모델 구조 (`MobileNetV4 Hybrid Large` + `DatologyAI/retr-opt-vit-b-32`)
- Teacher 로딩과 adaptive distillation 로직
- Offline teacher feature 검증 방식
- 학습 / 평가 / export / QAI Hub 흐름

---

## References

- [MobileNetV4](https://arxiv.org/abs/2404.10518) — Google Research / TensorFlow Models
- [DatologyAI retr-opt-vit-b-32](https://huggingface.co/DatologyAI/retr-opt-vit-b-32) — DatologyAI / Apache-2.0 retrieval text tower
- [SigLIP 2](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md) — Google Research
- [PE-Core / Perception Models](https://github.com/facebookresearch/perception_models) — Meta FAIR
- [OpenCLIP](https://github.com/mlfoundations/open_clip) — MLFoundations
- [TinyCLIP](https://arxiv.org/abs/2309.12314)

---

## License

프로젝트 코드는 Apache License 2.0을 따릅니다.

- 저장소 라이선스: [LICENSE](LICENSE)
- 외부 구성요소: [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)

> 외부 모델 가중치와 데이터셋은 별도 라이선스를 가질 수 있습니다.
> 재배포 또는 상업 사용 전 upstream 모델 카드와 데이터셋 약관을 반드시 재확인하세요.

---

## Acknowledgements

- Ross Wightman / timm
- Google Research / DeepMind
- Meta FAIR
- OpenCLIP contributors
- LPCVC 2026 organizers

