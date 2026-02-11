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

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Dataset Setup

MS-COCO 데이터셋을 다운로드하고 `config.yaml`의 경로를 수정합니다:

```yaml
data:
  coco_root: ./dataset/coco
  image_root: ./dataset/coco
```

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
│   └── export_onnx_split.py  # ONNX 변환
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
