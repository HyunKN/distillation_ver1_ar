# 🚀 MobileCLIP2 Retrieval Optimization

> LPCVC 2026을 위한 경량 이미지-텍스트 검색 모델 최적화

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## 📖 Overview

Apple의 **MobileCLIP2-S4**를 SOTA Teacher 모델(SigLIP 2, MetaCLIP 2 Worldwide)로 증류하여 모바일에서 빠르게 동작하는 이미지-텍스트 검색 AI를 학습합니다.

**핵심 특징:**
- 🎓 **Dual Teacher Distillation**: SigLIP 2 Giant + MetaCLIP 2 Worldwide (H/14, OpenCLIP 기준 약 7.44GB)
- ⚡ **Mobile-first**: 스마트폰에서 ~5ms 추론
- 📱 **50MB 미만**: LPCVC 대회 규격 준수
- 🔧 **최신 기술**: Mixed Precision, EMA, Gradient Clipping
- ℹ️ 라이선스/대회 규정 검토 결과에 따라 Student 모델(MobileCLIP2-S4)은 추후 OSI 호환 대체 모델로 변경될 수 있습니다.

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
│              ┌────────────────────────┐                │
│              │   MobileCLIP2-S4       │                │
│              │   (50MB, Student)      │                │
│              └────────────────────────┘                │
└─────────────────────────────────────────────────────────┘
```

### 💡 직관적인 모델 및 구조 설명

본 프로젝트는 고성능 이미지 검색 AI를 스마트폰 환경에 맞게 최적화하기 위해, **경량 학생 모델(Student)**과 압도적인 성능을 가진 두 개의 **선생님 모델(Teachers)**을 활용하는 **지식 증류(Knowledge Distillation)** 기법을 사용합니다.

#### 📱 1. 학생 모델: MobileCLIP2-S4
실제로 스마트폰 등 온디바이스(On-device) 환경에 배포되어 동작하는 소형 딥러닝 모델입니다.
* **선택 이유:** 수 GB에 달하는 거대한 모델들은 모바일 기기에서 구동이 불가능하거나 극도로 느립니다. MobileCLIP2-S4는 용량이 단 50MB에 불과하면서도 스마트폰에서 5ms 내외의 초고속 추론이 가능하여, LPCVC 대회의 빡빡한 하드웨어 제한(경량, 저지연)에 완벽히 부합합니다.

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

기본값은 COCO 모드(`data.mode: coco`)입니다.  
`D:\LPCVC_Data`의 원본 4개 데이터셋(coco/flickr30k/open_images/wit)을 함께 쓰려면, 먼저 아래 스크립트로 프로젝트 JSONL 포맷으로 변환하세요.

```bash
# 예시: Windows CMD
python scripts\preprocess\parse_lpcvc_sources.py --data_root D:\LPCVC_Data --out_dir dataset\lpcvc --val_ratio 0.01
```

전량 변환 전에 샘플 검증(권장):

```bash
python scripts\preprocess\parse_lpcvc_sources.py --data_root D:\LPCVC_Data --sample_per_source 5 --show_examples 12 --dry_run
```

용량 제한 서버용(예: 50GB) 축소 파싱 예시:

```bash
# coco/flickr30k/wit는 전량 유지, open_images만 60,000장으로 제한
python scripts\preprocess\parse_lpcvc_sources.py --data_root D:\LPCVC_Data --out_dir D:\LPCVC_Data\prepared_jsonl_50gb --source_caps open_images=60000 --val_ratio 0.01
```

운영 메모:
- 현재는 용량 제약으로 축소셋(약 18.8만 장)을 사용합니다.
- 추후 충분한 용량의 서버에서는 **약 30만 장 규모 데이터셋**으로 확장 학습할 예정입니다.

변환 후 `config.yaml`의 `data` 섹션을 아래처럼 바꿉니다:

```yaml
data:
  mode: jsonl
  image_root: D:/LPCVC_Data
  train_jsonl: ./dataset/lpcvc/train.jsonl
  val_jsonl: ./dataset/lpcvc/val.jsonl
```

참고: 전처리 스크립트는 `scripts/preprocess/` 폴더로 분리되어 있습니다.

### 3. Training

```bash
python run_train.py --config config.yaml
```

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
│   ├── model.py              # 모델 정의
│   ├── mobileclip2.py        # MobileCLIP2 로딩
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

## 📊 Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| R@1 | >25% | Top-1 Recall |
| R@5 | >50% | Top-5 Recall |
| R@10 | >60% | Top-10 Recall |
| Latency | <10ms | 모바일 추론 시간 |
| Model Size | <50MB | ONNX 파일 크기 |

## 🔬 Technical Details

자세한 기술 설명은 [PROJECT_GUIDE.md](./PROJECT_GUIDE.md)를 참조하세요:
- 모델 아키텍처 상세
- 학습 알고리즘 설명
- 적용된 기술 및 논문 출처

## 📚 References

- [MobileCLIP2](https://github.com/apple/ml-mobileclip) - Apple
- [SigLIP 2](https://arxiv.org/abs/2502.14786) - Google DeepMind
- [MetaCLIP 2](https://github.com/facebookresearch/MetaCLIP) - Meta AI
- [Meta CLIP 2 Collection](https://huggingface.co/collections/facebook/meta-clip-2) - Hugging Face
- [TinyCLIP](https://arxiv.org/abs/2309.12314) - Affinity Distillation

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgements

- Apple for MobileCLIP2
- Google DeepMind for SigLIP 2
- Meta AI for MetaCLIP 2
- LPCVC 2026 Organizers
