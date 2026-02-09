# 🚀 MobileCLIP2 Retrieval Optimization

> LPCVC 2026을 위한 경량 이미지-텍스트 검색 모델 최적화

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## 📖 Overview

Apple의 **MobileCLIP2-S4**를 SOTA Teacher 모델(SigLIP 2, MetaCLIP 2)로 증류하여 모바일에서 빠르게 동작하는 이미지-텍스트 검색 AI를 학습합니다.

**핵심 특징:**
- 🎓 **Dual Teacher Distillation**: SigLIP 2 Giant + MetaCLIP 2 Huge
- ⚡ **Mobile-first**: 스마트폰에서 ~5ms 추론
- 📱 **50MB 미만**: LPCVC 대회 규격 준수
- 🔧 **최신 기술**: Mixed Precision, EMA, Gradient Clipping

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Teacher Models                        │
│  ┌─────────────────────┐  ┌─────────────────────────┐  │
│  │   SigLIP 2 Giant    │  │    MetaCLIP 2 Huge       │  │
│  │   (7.5GB, 60%)      │  │    (2.5GB, 40%)          │  │
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
python run_train.py
```

### 4. Evaluation

```bash
python scripts/eval.py --checkpoint checkpoints/best.pt
```

### 5. Export to ONNX

```bash
python scripts/export_onnx_split.py --checkpoint checkpoints/best.pt
```

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
│   └── loss.py               # 손실 함수
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
- [TinyCLIP](https://arxiv.org/abs/2309.12314) - Affinity Distillation

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgements

- Apple for MobileCLIP2
- Google DeepMind for SigLIP 2
- Meta AI for MetaCLIP 2
- LPCVC 2026 Organizers
