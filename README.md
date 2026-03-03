# 🚀 MobileCLIP2 Retrieval Optimization

> LPCVC 2026을 위한 Apple MobileCLIP2-S0 기반 모바일 최적화 멀티모달 검색 모델

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## 📖 Overview

Apple의 **MobileCLIP2-S0** (이미지+텍스트 사전 학습 완성형 모델)을 학생 모델로 채택하고, SOTA Teacher 모델(SigLIP 2 Giant, MetaCLIP 2)로 지식 증류(Distillation)하여 모바일에서 빠르고 정확하게 동작하는 이미지-텍스트 검색 AI를 구축합니다.

**핵심 특징:**
- 🥇 **Multi-Modal Pre-trained Student**: 이미지-텍스트 매칭이 이미 사전 학습된 MobileCLIP2-S0 (11.4M Vision / 63.4M Text)
- 🎓 **Dual Teacher Distillation**: SigLIP 2 Giant + MetaCLIP 2 Worldwide (H/14)
- ⚡ **Mobile-first**: XR2 Gen 2 기준 Image 9.1ms / Text 3.5ms 실측
- 🔧 **최신 기술**: Mixed Precision, EMA, Gradient Clipping, torch.compile

## 📚 문서 역할 분담

- `README.md`: 프로젝트 흐름, 핵심 기술, 실행 방법, 설정값 조작(실행 중심)
- `PROJECT_GUIDE.md`: 코드/알고리즘/아키텍처/라이선스/논문 근거(심화 설명)
- 원칙: 둘 다 출처를 명시하며, README는 요약/실행 관점으로 유지합니다.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Teacher Models                        │
│  ┌─────────────────────┐  ┌─────────────────────────┐  │
│  │   SigLIP 2 Giant    │  │ MetaCLIP 2 Worldwide H14 │  │
│  │   (7.5GB, 60%)      │  │   (~7.44GB, 40%)         │  │
│  └──────────┬──────────┘  └───────────┬─────────────┘  │
│             │     Knowledge Distillation     │          │
│             └─────────────┬─────────────────┘          │
│                           ▼                             │
│          ┌──────────────────────────────┐              │
│          │  MobileCLIP2-S0 (Apple)      │              │
│          │  V: 11.4M (9.1ms) Image Enc │              │
│          │  T: 63.4M (3.5ms) Text Enc  │              │
│          │  Pre-trained Multi-Modal     │              │
│          └──────────────────────────────┘              │
└─────────────────────────────────────────────────────────┘
```

### 💡 직관적인 모델 및 구조 설명

본 프로젝트는 고성능 이미지 검색 AI를 스마트폰 환경에 맞게 최적화하기 위해, **경량 학생 모델(Student)**과 압도적인 성능을 가진 두 개의 **선생님 모델(Teachers)**을 활용하는 **지식 증류(Knowledge Distillation)** 기법을 사용합니다.

#### 📱 1. 학생 모델: Apple MobileCLIP2-S0
실제로 스마트폰/XR 디바이스에서 작동할 초경량 멀티모달 모델입니다.
* **비전 인코더 (11.4M params):** Apple이 모바일 NPU에 최적화하여 설계한 MobileOne 기반 아키텍처. XR2 Gen 2에서 **9.1ms** 실측.
* **텍스트 인코더 (63.4M params):** 이미지-텍스트 매칭이 사전 학습된 내장 텍스트 인코더. XR2 Gen 2에서 **3.5ms** 실측.
* **선택 이유:** 이미지와 텍스트를 연결하는 멀티모달 지식이 **이미 사전 학습 완료**된 상태이므로, 다른 조처럼 빈 깡통 부품(FastViT 등)을 조립하여 처음부터 가르칠 필요 없이 즉시 파인튜닝이 가능합니다. 동급 파라미터(11M) 대비 압도적인 Recall 성능을 발휘합니다.

#### 🔍 2. 첫 번째 선생님: SigLIP 2 Giant
학생 모델에게 **정밀한 텍스트-이미지 매칭 능력**을 전수해 줄 첫 번째 전문가 모델입니다.
* **선택 이유:** SigLIP 2는 현재 업계 최고 수준(SOTA)의 비전-언어 모델 중 하나로, 이미지의 미세한 디테일과 복잡한 언어적 문맥을 연결하는 데 탁월한 성능을 보입니다. 작은 학생 모델이 놓치기 쉬운 세밀한 특징들을 스스로 찾아내도록 지도합니다.

#### 🌐 3. 두 번째 선생님: MetaCLIP 2 Worldwide
학생 모델에게 **방대한 글로벌 상식**을 전수해 줄 두 번째 전문가 모델입니다.
* **선택 이유:** 인터넷 상의 어마어마한 양의 전 세계(Worldwide) 데이터를 바탕으로 학습되었기 때문에, 현존하는 가장 넓은 범용 시각 지식을 보유하고 있습니다. 학생 모델이 다양한 환경이나 낯선 이미지 패턴에서도 당황하지 않고 유연하게 대응할 수 있도록 시야를 넓혀줍니다.

---

#### ⚖️ 왜 두 개의 선생님 모델(Dual Teacher)을 사용하나요?
단일 선생님 모델에만 의존할 경우 해당 모델이 가진 '특정 학습 편향'까지 학생이 그대로 물려받을 위험이 있습니다. 
마치 전공 분야가 다른 두 명의 명문대 교수님에게 수업을 듣는 것처럼, **정밀한 묘사에 강한 SigLIP 2**와 **방대한 일반 상식에 강한 MetaCLIP 2**를 결합하면 각각의 장점만을 안전하게 취할 수 있습니다. 두 선생님의 예측값(Soft-targets)을 6:4 비율로 융합하여 학생에게 전달함으로써, 결과적으로 학생 모델은 물리적인 크기는 작으면서도 다방면으로 뛰어난 시각 지능을 갖추게 됩니다.

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Dataset Setup

기본값은 JSONL 모드(`data.mode: jsonl`)입니다.  
서버에서 프로젝트 루트 아래 `dataset/` 폴더를 사용할 때, 원본 4개 데이터셋(coco/flickr30k/open_images/wit)을 먼저 JSONL로 변환하세요.

```bash
# 예시: Windows CMD (프로젝트 루트에 dataset/가 있는 경우)
python scripts\preprocess\parse_lpcvc_sources.py --data_root dataset --out_dir dataset\prepared_jsonl --val_ratio 0.01
```

전량 변환 전에 샘플 검증(권장):

```bash
python scripts\preprocess\parse_lpcvc_sources.py --data_root dataset --sample_per_source 5 --show_examples 12 --dry_run
```

용량 제한 서버용(예: 50GB) 축소 파싱 예시:

```bash
# coco/flickr30k/wit는 전량 유지, open_images만 60,000장으로 제한
python scripts\preprocess\parse_lpcvc_sources.py --data_root dataset --out_dir dataset\prepared_jsonl_50gb --source_caps open_images=60000 --val_ratio 0.01
```

운영 메모:
- 현재는 용량 제약으로 축소셋(약 18.8만 장)을 사용합니다.
- 추후 충분한 용량의 서버에서는 **약 30만 장 규모 데이터셋**으로 확장 학습할 예정입니다.

#### 📊 내 데이터셋 구조 및 비율

**[이미지 기준(중복 제거)]**
- **coco:** 123,287장 (`train2017` 118,287 + `val2017` 5,000)
- **flickr30k:** 31,783장
- **open_images:** 125,436장
- **wit:** 2,710장
- **합계:** **283,216장**

참고:
- JSONL의 `train/val` 행 수는 "이미지 수"와 다를 수 있습니다(한 이미지에 여러 캡션이 붙기 때문).

**[비율 (중복 제거 가정 총 283,216장 기준)]**
- **open_images:** 125,436장 (44.29%)
- **coco:** 123,287장 (43.53%)
- **flickr30k:** 31,783장 (11.22%)
- **wit:** 2,710장 (0.96%)

**[메타데이터 샘플 수 (jsonl)]**
- `prepared_jsonl`: train 275,212 / val 2,777 (총 277,989)
- `prepared_jsonl_40gb`: train 185,678 / val 1,873
- `prepared_jsonl_witcheck`: train 2,457 / val 24

변환 후 `config.yaml`의 `data` 섹션을 아래처럼 바꿉니다:

```yaml
data:
  mode: jsonl
  image_root: dataset
  train_jsonl: dataset/prepared_jsonl/train.jsonl
  val_jsonl: dataset/prepared_jsonl/val.jsonl
```

참고: 전처리 스크립트는 `scripts/preprocess/` 폴더로 분리되어 있습니다.

### 2.5 Offline Teacher Feature Extraction (권장)

처음 실행은 아래 2단계로 진행하면 됩니다.

1. `run_train.py` 전에 Teacher feature를 1회 추출
이유:
- Teacher를 학습 루프에서 매번 돌리지 않아서 VRAM/학습시간을 크게 줄입니다.
- 이후 실험을 반복해도 같은 Teacher feature를 재사용할 수 있습니다.

명령어:
```bash
# Windows CMD
python scripts\extract_features.py --config config.yaml --out_dir features --override data.train_augment=false

# Linux/macOS
python scripts/extract_features.py --config config.yaml --out_dir features --override data.train_augment=false
```

추출 후 만들어져야 하는 파일(예):
- `features/teacher_0_train.pt`
- `features/teacher_1_train.pt`
- `features/teacher_0_val.pt`
- `features/teacher_1_val.pt`

2. 생성된 feature 경로를 config에 연결한 뒤 `run_train.py` 실행
이유:
- 학습기가 이 경로를 보고 Teacher feature를 읽어 distillation을 수행합니다.

`config.yaml` 설정:
```yaml
distill:
  offline_feature_dir: features
```

학습 실행:
```bash
python run_train.py --config config.yaml
```

실행 중 확인 포인트:
1. feature 추출 로그에 `Effective data.train_augment: False`가 출력되는지
2. 학습 로그에 `[Train] Offline mode — Teacher NOT loaded.`가 출력되는지

### 3. Training

```bash
python run_train.py --config config.yaml
```

설정값을 세밀하게 바꿔 실행하려면 `--override key=value`를 여러 번 사용합니다.

```bash
# 예시 1) 스모크 테스트 (빠른 동작 확인)
python run_train.py --config config.yaml --override train.epochs=1 --override data.batch_size=32

# 예시 2) 학습률/EMA/Distill 강도 조정
python run_train.py --config config.yaml --override train.lr=1e-4 --override train.use_ema=true --override loss.w_distill_affinity=0.3

# 예시 3) Teacher OFF로 파이프라인 검증
python run_train.py --config config.yaml --override distill.use_teacher=false
```

설정 조작 기준:
- 정적 기준값은 `config.yaml`에 기록합니다.
- 실험 파라미터는 `--override`로 바꾸고 실행 로그에 남깁니다.
- 재현이 필요한 실험은 사용한 override 목록을 별도 메모/커밋 메시지에 저장합니다.

학습 전 확인:
1. `distill.offline_feature_dir`가 실제 `.pt` feature 경로와 일치하는지
2. `data.mode`/`train_jsonl`/`val_jsonl`이 현재 데이터셋 구조와 일치하는지
3. 로그에 의도한 override가 출력되는지 (`[Config] Overrides:`)

### 4. Evaluation

```bash
# Linux/macOS
PYTHONPATH=src python scripts/eval.py --config config.yaml --ckpt runs/lpcvc_clip_lite/best.pt

# Windows CMD
set PYTHONPATH=src && python scripts\eval.py --config config.yaml --ckpt runs\lpcvc_clip_lite\best.pt
```

### 5. Export to ONNX

```bash
# Linux/macOS
PYTHONPATH=src python scripts/export_onnx_split.py --config config.yaml --ckpt runs/lpcvc_clip_lite/best.pt --out_dir exported_onnx

# Windows CMD
set PYTHONPATH=src && python scripts\export_onnx_split.py --config config.yaml --ckpt runs\lpcvc_clip_lite\best.pt --out_dir exported_onnx
```

### 6. Qualcomm AI Hub 업로드/컴파일/프로파일

```bash
# 1) 최초 1회 인증
qai-hub configure --api_token <YOUR_QAI_HUB_TOKEN>

# 2) ONNX 업로드 + 컴파일 + 프로파일
python compile_and_profile.py --onnx_dir exported_onnx --img_name image_encoder.onnx --txt_name text_encoder.onnx --device "XR2 Gen 2 (Proxy)"
```

컴파일만 먼저 하고 싶으면 `--skip_profile`를 사용합니다.

```bash
python compile_and_profile.py --onnx_dir exported_onnx --img_name image_encoder.onnx --txt_name text_encoder.onnx --device "XR2 Gen 2 (Proxy)" --skip_profile
```

이미 컴파일된 Job ID가 있으면, 재컴파일 없이 프로파일만 따로 제출할 수 있습니다 (Windows CMD 한 줄):

```bash
python -c "import qai_hub as hub; d=hub.Device('XR2 Gen 2 (Proxy)'); m=hub.get_job('<IMAGE_COMPILE_JOB_ID>').get_target_model(); p=hub.submit_profile_job(model=m, device=d, options='--max_profiler_iterations 100'); print('image', p.job_id); m=hub.get_job('<TEXT_COMPILE_JOB_ID>').get_target_model(); p=hub.submit_profile_job(model=m, device=d, options='--max_profiler_iterations 100'); print('text', p.job_id)"
```

참고:
- `--skip_profile`로 실행하면 PROFILE 탭에는 아무것도 생기지 않고 COMPILE 탭에만 생성됩니다.
- `image_encoder.onnx.data`, `text_encoder.onnx.data`는 ONNX 외부 가중치 파일이며 `.onnx`와 같은 폴더에 두면 업로드 시 함께 처리됩니다.

## 📁 Project Structure

```
├── run_train.py              # 학습 시작점
├── config.yaml               # 설정 파일
├── src/lpcvc_retrieval/      # 핵심 코드
│   ├── train.py              # 학습 로직
│   ├── model.py              # 모델 팩토리
│   ├── mobileclip2.py        # MobileCLIP2-S0 학생 모델 래퍼
│   ├── distill.py            # Teacher 모델 & Distillation
│   ├── data.py               # 데이터 로딩
│   └── losses.py             # 손실 함수
├── scripts/
│   ├── eval.py               # 성능 평가
│   ├── export_onnx_split.py  # ONNX 변환
│   └── preprocess/
│       ├── parse_lpcvc_sources.py        # LPCVC 원본 4종(coco/flickr/open_images/wit) 파싱
│       └── materialize_upload_subset.py  # 업로드용 subset 데이터 구성
└── compile_and_profile.py    # Qualcomm AI Hub
```

## ⚙️ Configuration

주요 설정 (`config.yaml`):

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `batch_size` | 128 | GPU 메모리에 따라 조정 |
| `lr` | 5e-4 | 학습률 |
| `epochs` | 10 | 학습 에폭 |
| `use_teacher` | true | Teacher Distillation 사용 |
| `amp` | true | Mixed Precision 학습 |

## 📊 Performance Targets & Profiling (MobileCLIP2-S0)

| Metric | Target | Current (XR2 Gen 2) | Description |
|--------|--------|---------------------|-------------|
| R@1 | >25% | - | Top-1 Recall |
| R@5 | >50% | - | Top-5 Recall |
| R@10 | >60% | - | Top-10 Recall |
| Image Latency | <10ms | **9.1 ms** | 비전 인코더 단독 추론 (채점 기준) |
| Text Latency | - | 3.5 ms | 텍스트 인코더 단독 추론 (사전 연산용) |
| Image Params | - | **11.4 M** | 비전 인코더 모델 파라미터 수 |
| Text Params | - | 63.4 M | 텍스트 인코더 모델 파라미터 수 |
| Image Size (ONNX) | <50MB | 43.7 MB | 비전 인코더 모델 용량 |

측정 출처:
- Image/Text latency는 Qualcomm AI Hub `XR2 Gen 2 (Proxy)` 프로파일 결과(내부 실험 로그) 기준입니다.
- 모델 파라미터/용량은 모델 카드 및 ONNX export 산출물 기준입니다.

## 🔬 Technical Details

자세한 기술 설명은 [PROJECT_GUIDE.md](./PROJECT_GUIDE.md)를 참조하세요:
- 모델 아키텍처 상세
- 학습 알고리즘 설명
- 적용된 기술 및 논문 출처
- 코드 단위 동작 설명 및 라이선스 검토

## 📚 References

- [MobileCLIP2](https://github.com/apple/ml-mobileclip) - Apple ML Research (Student)
- [SigLIP 2](https://arxiv.org/abs/2502.14786) - Google DeepMind (Teacher 1)
- [MetaCLIP 2](https://github.com/facebookresearch/MetaCLIP) - Meta AI (Teacher 2)
- [Meta CLIP 2 Collection](https://huggingface.co/collections/facebook/meta-clip-2) - Hugging Face
- [TinyCLIP](https://arxiv.org/abs/2309.12314) - Affinity Distillation
- [OpenCLIP](https://github.com/mlfoundations/open_clip) - Model Loading Framework

## 📝 License

This project is licensed under the MIT License.

### Third-Party Model Licenses

| 구분 | 라이선스 | 비고 |
|---|---|---|
| **프로젝트 코드** | MIT | 본 저장소 코드 |
| **MobileCLIP2 (Student)** | Code: MIT / Weights: Apple ML Research TOU | [모델 카드 확인](https://huggingface.co/apple/MobileCLIP2-S0) |
| **SigLIP 2 Giant (Teacher 1)** | Apache 2.0 | [모델 카드 확인](https://huggingface.co/timm/ViT-gopt-16-SigLIP2-256) |
| **MetaCLIP 2 Worldwide (Teacher 2)** | CC-BY-NC-4.0 | 비상업 조항 포함. [모델 카드 확인](https://huggingface.co/facebook/metaclip-2-worldwide-huge-quickgelu) |
| **OpenCLIP 프레임워크** | MIT | [저장소](https://github.com/mlfoundations/open_clip) |

> ⚠️ 모델 가중치 재배포/상업 사용 전 각 모델 카드의 라이선스 원문을 반드시 재확인하세요.

## 🙏 Acknowledgements

- Apple for MobileCLIP2
- Google DeepMind for SigLIP 2
- Meta AI for MetaCLIP 2
- LPCVC 2026 Organizers
