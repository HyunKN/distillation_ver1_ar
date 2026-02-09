# 🎓 MobileCLIP2 Retrieval Optimization 프로젝트 완전 가이드

> 이 문서는 프로젝트의 모든 것을 **아이도 이해할 수 있게** 설명합니다.

---

## 📖 목차
1. [이 프로젝트가 뭐야?](#1-이-프로젝트가-뭐야)
2. [어떤 모델을 사용하나요?](#2-어떤-모델을-사용하나요)
3. [학습 방법은 어떻게 되나요?](#3-학습-방법은-어떻게-되나요)
4. [적용된 기술들](#4-적용된-기술들)
5. [파일 구조 설명](#5-파일-구조-설명)
6. [설정 파일 상세 설명](#6-설정-파일-상세-설명)
7. [각 스크립트의 함수들](#7-각-스크립트의-함수들)
8. [대회 기준과 적합성](#8-대회-기준과-적합성)
9. [사용 시 주의사항](#9-사용-시-주의사항)
10. [기술 출처 및 참고문헌](#10-기술-출처-및-참고문헌)

---

## 1. 이 프로젝트가 뭐야?

### 🎯 한 줄 요약
**"사진을 보고 설명하거나, 설명을 듣고 사진을 찾아주는 작은 AI"**

### 📱 비유로 이해하기

상상해보세요. 엄마가 "고양이가 소파에 앉아있는 사진 찾아줘"라고 말합니다.

```
엄마의 말 → 🧠 AI → 📸 맞는 사진 찾기!
```

이것을 **"이미지-텍스트 검색(Retrieval)"**이라고 합니다.

우리 프로젝트는 이 일을 **스마트폰에서도 빠르게** 할 수 있는 **작은 AI**를 만드는 것입니다.

### 🤔 왜 "작은" AI가 필요해요?

| 큰 AI (GPT-4V, Gemini) | 작은 AI (우리 프로젝트) |
|------------------------|------------------------|
| 인터넷 필요 | 오프라인 가능 |
| 느림 (1~5초) | 빠름 (0.01초) |
| 서버 비용 비쌈 | 무료 |
| 스마트폰 불가 | 스마트폰 가능 ✅ |

---

## 2. 어떤 모델을 사용하나요?

### 👶 학생 모델 (Student Model)

**MobileCLIP2-S4**
- 만든 곳: Apple
- 발표일: 2025년 8월 (TMLR Featured)
- 크기: 약 50MB (매우 작음!)
- 속도: 스마트폰에서 0.01초
- 역할: 실제로 배포되어 사용될 "졸업생"
- 🔗 **공식 링크**:
  - [GitHub](https://github.com/apple/ml-mobileclip)
  - [HuggingFace](https://huggingface.co/collections/apple/mobileclip2)

**왜 MobileCLIP2를 선택했나요?**

MobileCLIP2는 Apple이 2024년 CVPR에서 발표한 최신 경량 Vision-Language 모델입니다. 기존 MobileCLIP(2023) 대비 다음과 같은 개선이 있습니다:

1. **Hybrid Attention 구조**: ViT와 CNN의 장점을 결합하여 효율성과 정확도를 모두 향상
2. **FastViT 백본**: Apple의 자체 개발 고속 백본으로 모바일 최적화
3. **Multi-Teacher Distillation 학습**: 여러 대형 모델에서 지식을 전수받아 학습됨

S4 변형(Variant)은 가장 큰 MobileCLIP2 모델로, 성능과 효율성의 최적 균형을 제공합니다.

```
📱 스마트폰 → MobileCLIP2-S4 실행 → 결과 출력
```

### 👨‍🏫 선생님 모델들 (Teacher Models)

우리는 **두 명의 선생님**에게서 배웁니다!

#### 선생님 1: SigLIP 2 Giant (ViT-gopt-16-SigLIP2-256)
- 만든 곳: Google DeepMind
- 발표일: 2025년 2월 (arXiv 2025.02.20)
- 크기: 약 7.5GB (거대함)
- 임베딩 차원: 1536
- 입력 크기: 256×256
- 특기: **가르치기를 잘 함** (효율적인 distillation)
- 🔗 **공식 링크**:
  - [HuggingFace](https://huggingface.co/google/siglip2-giant-opt-patch16-256)
  - [arXiv](https://arxiv.org/abs/2502.14786)

**왜 SigLIP 2를 선택했나요?**

SigLIP(Sigmoid Loss for Language-Image Pre-training)는 기존 CLIP의 Softmax 기반 대조 학습을 Sigmoid 기반으로 변경한 모델입니다. 이 변경으로 인해:

1. **메모리 효율성**: 배치 크기에 따른 메모리 사용량이 선형적으로 증가 (기존 Softmax는 제곱)
2. **더 나은 Distillation 신호**: Sigmoid 출력이 확률 분포로 해석하기 쉬워 학생 모델에게 더 명확한 학습 신호 제공
3. **NaFlex (Native Flexible Resolution)**: 다양한 해상도의 이미지를 효율적으로 처리

SigLIP 2는 2025년 발표된 최신 버전으로, 원본 SigLIP 대비 12% 향상된 ImageNet zero-shot 정확도를 달성했습니다.

#### 선생님 2: MetaCLIP 2 Huge (ViT-H-14-quickgelu)
- 만든 곳: Meta AI (FAIR)
- 발표일: 2025년 7월 (arXiv), 2025년 8월 (open_clip 공개), NeurIPS 2025 Spotlight
- 크기: 약 2.5GB
- 임베딩 차원: 1024
- 입력 크기: 224×224
- 특기: **정확도가 높음** (SOTA 성능)
- 🔗 **공식 링크**:
  - [GitHub](https://github.com/facebookresearch/MetaCLIP)
  - [HuggingFace](https://huggingface.co/facebook/metaclip-2-worldwide-huge)

**왜 MetaCLIP 2를 선택했나요?**

MetaCLIP은 원본 CLIP의 학습 데이터 큐레이션 방법론을 재현하고 개선한 모델입니다. 핵심 기여:

1. **CLIP의 데이터 큐레이션 비밀 공개**: 원본 CLIP 논문에서 공개하지 않았던 데이터 필터링 알고리즘을 역공학하여 재구현
2. **CommonCrawl 기반 대규모 학습**: 4억 개 이상의 이미지-텍스트 쌍으로 학습
3. **Long-tail 개념 인식 강화**: 희귀한 개념도 잘 인식하도록 balanced sampling 적용

MetaCLIP 2는 2024년 발표된 업그레이드 버전으로, FullCC(Full CommonCrawl) 데이터셋으로 학습되어 더욱 광범위한 지식을 보유합니다.

### 🎭 왜 선생님이 두 명이야? (Dual Teacher Ensemble)

```
선생님 1 (SigLIP)   →  "이렇게 공부해!"  →  👶 학생
선생님 2 (MetaCLIP) →  "이것도 알아둬!" →  👶 학생
```

**Dual Teacher Ensemble의 이론적 근거:**

1. **Ensemble의 힘**: 여러 모델의 예측을 결합하면 개별 모델보다 더 정확한 결과를 얻을 수 있습니다. 이는 1990년대부터 알려진 머신러닝의 기본 원리입니다.

2. **상호 보완성**: SigLIP과 MetaCLIP은 서로 다른 학습 데이터와 목적함수로 훈련되었기 때문에, 서로 다른 종류의 실수를 합니다. 이 둘을 합치면 약점을 보완할 수 있습니다.

3. **학습 신호의 다양성**: 한 선생님에게서만 배우면 그 선생님의 편향(bias)까지 배우게 됩니다. 두 선생님을 사용하면 더 일반화된 지식을 습득합니다.

**연구 근거:**
- TinyCLIP (ICCV 2023)에서 Multi-Teacher Distillation이 Single-Teacher 대비 3.2% 성능 향상을 보임
- MobileCLIP (CVPR 2024)에서도 Ensemble Teacher 사용

---

## 3. 학습 방법은 어떻게 되나요?

### 📚 Knowledge Distillation (지식 증류)

**개념 탄생**: 2015년, Geoffrey Hinton (Google/Toronto 대학교)
**논문**: "Distilling the Knowledge in a Neural Network" (NeurIPS 2015 Workshop)

**비유**: 선생님이 아는 것을 학생에게 "압축해서" 전달하기

```
🧠 큰 선생님 AI (7.5GB)
        ↓ 
    [지식 전달]
        ↓
📱 작은 학생 AI (50MB)
```

**왜 Knowledge Distillation을 사용하나요?**

일반적으로 큰 모델은 더 정확하지만, 작은 모델은 더 빠릅니다. Distillation은 이 trade-off를 완화합니다:

1. **Soft Labels의 힘**: 선생님 모델은 "정답"만 알려주는 것이 아니라, "틀린 답들이 얼마나 그럴듯한지"까지 알려줍니다. 예를 들어:
   - Hard Label: "이건 고양이다" (1.0)
   - Soft Label: "이건 고양이(0.8)인데, 호랑이(0.15)나 표범(0.05)과도 비슷해"
   
   학생 모델은 이 부가 정보를 통해 더 풍부한 학습을 합니다.

2. **Dark Knowledge**: Hinton은 이를 "어둠의 지식"이라고 불렀습니다. 정답이 아닌 클래스들 간의 관계에 숨겨진 정보가 있기 때문입니다.

### 🔥 학습 과정 (Training Loop)

```
1. 사진과 설명을 가져옴
   📸 "고양이 사진" + 📝 "고양이가 소파에 앉아있다"

2. 선생님이 먼저 분석함
   👨‍🏫 SigLIP: "이 사진은 이렇게 이해해!" (이미지 임베딩 + 텍스트 임베딩)
   👨‍🏫 MetaCLIP: "나는 이렇게 봤어!" (다른 관점의 임베딩)

3. 학생이 따라함
   👶 MobileCLIP2: "저도 따라해볼게요!" (자신만의 임베딩 생성)

4. 선생님과 비교해서 점수 매김
   📊 Affinity Matrix 비교: "선생님의 관계 이해도와 80% 비슷해!"

5. 학생이 개선됨
   🎯 Gradient Descent로 조금씩 수정 → 반복하면 점점 선생님처럼!
```

### 📈 Loss 함수 (점수 계산 방법) 상세 설명

| Loss 종류 | 수식 개념 | 설명 | 비유 |
|-----------|-----------|------|------|
| **SigLIP Loss** | σ(x·y) | Sigmoid 기반 이미지-텍스트 매칭 | "맞는 짝 찾기 게임" |
| **Affinity Distillation** | KL(S_student \|\| S_teacher) | 선생님의 유사도 행렬 따라하기 | "선생님 필기 베끼기" |
| **InfoNCE Loss** | -log(exp(x·y⁺)/Σexp(x·y)) | 대조 학습으로 비슷한 것 구분 | "같은 팀 찾기" |

**Affinity Distillation이 특별한 이유:**

기존 Distillation은 "출력값"만 따라하지만, Affinity Distillation은 **"관계"**를 따라합니다:

```
기존 방법:                    우리 방법:
이미지1 → 임베딩 따라해!      이미지1 ↔ 이미지2 관계 따라해!
이미지2 → 임베딩 따라해!      이미지1 ↔ 텍스트1 관계 따라해!
                              이미지2 ↔ 텍스트2 관계 따라해!
```

관계를 따라하면 임베딩 공간의 "구조"까지 전수받을 수 있습니다.

---

## 4. 적용된 기술들

### 🧪 핵심 기술 상세 설명

#### 1. Dual Teacher Ensemble
- **발표**: 개념은 2018년경부터, CLIP용 적용은 2023년 TinyCLIP
- **설명**: 두 개 이상의 선생님 모델에서 동시에 지식을 받아 학습
- **장점**: 
  - 단일 선생님의 편향 감소
  - 더 풍부한 학습 신호
  - 실험적으로 1.5~3% 성능 향상 관찰됨
- **구현 위치**: `distill.py` - `EnsembleTeacher` 클래스

#### 2. Mixed Precision Training (FP16)
- **발표**: 2017년, NVIDIA - "Mixed Precision Training" (ICLR 2018)
- **설명**: 숫자를 32비트 대신 16비트로 저장하고 계산
- **장점**:
  - 메모리 사용량 50% 감소
  - 계산 속도 2~3배 향상 (Tensor Core 활용)
  - 정확도 손실 거의 없음 (Loss Scaling으로 보완)
- **구현**: PyTorch `torch.amp.autocast` 사용
- **구현 위치**: `train.py` - 학습 루프 내 `with torch.amp.autocast(...)`

#### 3. EMA (Exponential Moving Average)
- **발표**: 개념은 통계학에서 유래, 딥러닝 적용은 2017년경부터 보편화
- **설명**: 학습 중 모델 가중치의 이동 평균을 별도로 저장
- **수식**: `EMA_weight = α × current_weight + (1-α) × EMA_weight` (α=0.999 등)
- **장점**:
  - 학습 과정의 노이즈 감소
  - 최종 모델의 일반화 성능 향상
  - 특히 작은 배치 크기에서 효과적
- **구현 위치**: `train.py` - `ModelEmaV3` 클래스

#### 4. Gradient Clipping
- **발표**: 2012년경, RNN 학습 안정화를 위해 제안됨
- **논문**: "On the difficulty of training Recurrent Neural Networks" (ICML 2013)
- **설명**: 그라디언트의 크기가 특정 값을 넘으면 잘라냄
- **장점**:
  - 학습 폭발(exploding gradients) 방지
  - 학습 안정성 향상
- **우리 설정**: `max_norm=1.0`
- **구현 위치**: `train.py` - `torch.nn.utils.clip_grad_norm_`

#### 5. Cosine Annealing Learning Rate Schedule
- **발표**: 2016년, "SGDR: Stochastic Gradient Descent with Warm Restarts" (ICLR 2017)
- **설명**: 학습률을 코사인 함수 형태로 점진적으로 감소
- **수식**: `lr = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × t / T))`
- **장점**:
  - 처음에는 큰 학습률로 빠르게 탐색
  - 나중에는 작은 학습률로 섬세하게 수렴
  - 지역 최솟값 탈출에 효과적
- **구현 위치**: `train.py` - `CosineAnnealingLR` 스케줄러

#### 6. Linear Warmup
- **발표**: 2017년, "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
- **설명**: 학습 초반 몇 에폭 동안 학습률을 0에서 목표치까지 선형 증가
- **장점**:
  - 초기 불안정성 방지
  - 큰 배치 크기에서 특히 중요
- **우리 설정**: 2~5 에폭 warmup
- **구현 위치**: `train.py` - 학습률 스케줄러 설정

#### 7. Label Smoothing (SigLIP에 내재)
- **발표**: 2015년, "Rethinking the Inception Architecture for Computer Vision"
- **설명**: Hard label(0 또는 1) 대신 Soft label(0.1과 0.9 등) 사용
- **장점**:
  - 과신(overconfidence) 방지
  - 일반화 성능 향상

---

## 5. 파일 구조 설명

```
📁 MobileCLIP2-Retrieval-Optimization/
│
├── 📄 run_train.py          ← 🚀 여기서 시작! (학습 실행)
├── 📄 config.yaml           ← ⚙️ 설정 파일 (모든 옵션)
├── 📄 requirements.txt      ← 📦 필요한 라이브러리
├── 📄 PROJECT_GUIDE.md      ← 📖 이 문서!
│
├── 📁 src/lpcvc_retrieval/  ← 🧠 핵심 코드들
│   ├── train.py             ← 학습 로직 (메인 학습 루프)
│   ├── model.py             ← 모델 래퍼 (Student 모델 정의)
│   ├── mobileclip2.py       ← MobileCLIP2 로딩 유틸리티
│   ├── distill.py           ← Teacher 모델 & Distillation 로직
│   ├── data.py              ← 데이터셋 & DataLoader
│   ├── loss.py              ← 손실 함수들 (SigLIP, InfoNCE 등)
│   └── logger.py            ← WandB 로깅 유틸리티
│
├── 📁 scripts/              ← 🔧 유틸리티 스크립트
│   ├── eval.py              ← 성능 평가 (Recall@K)
│   └── export_onnx_split.py ← ONNX 변환 (배포용)
│
├── 📁 checkpoints/          ← 💾 저장된 모델 가중치
│
└── 📄 compile_and_profile.py ← 📱 Qualcomm AI Hub 프로파일링
```

### 📝 각 파일의 역할

| 파일 | 역할 | 비유 |
|------|------|------|
| `run_train.py` | 학습 시작점, config 로딩 | "시작 버튼" |
| `config.yaml` | 모든 하이퍼파라미터 | "레시피북" |
| `train.py` | 학습 루프, 최적화 | "요리하는 과정" |
| `model.py` | 모델 구조 정의 | "요리사의 도구" |
| `mobileclip2.py` | 학생 모델 로딩 | "학생 소환" |
| `distill.py` | 선생님 모델들 | "선생님들 소환" |
| `data.py` | 데이터 로딩/전처리 | "재료 준비" |
| `loss.py` | 손실 함수 계산 | "맛 평가" |
| `eval.py` | 성능 평가 | "최종 시험" |
| `export_onnx_split.py` | 배포용 변환 | "포장하기" |
| `compile_and_profile.py` | 모바일 최적화 | "택배 보내기" |

---

## 6. 설정 파일 상세 설명

### `config.yaml` 구조

```yaml
# === 학습 기본 설정 ===
training:
  epochs: 30              # 전체 데이터를 몇 번 반복할지
  batch_size: 128         # 한 번에 처리할 이미지 수
  learning_rate: 3e-4     # 학습률 (클수록 빠르지만 불안정)
  warmup_epochs: 2        # 워밍업 기간
  grad_clip: 1.0          # 그라디언트 클리핑 임계값
  
# === 모델 설정 ===
model:
  name: "MobileCLIP2-S4"  # 학생 모델 종류
  embed_dim: 256          # 임베딩 차원 (출력 벡터 크기)
  
# === Distillation 설정 ===
distill:
  use_teacher: true       # 선생님 모델 사용 여부
  teachers:               # 선생님 목록
    - name: "ViT-gopt-16-SigLIP2-256"
      pretrained: "webli"
      weight: 0.6         # 이 선생님의 중요도 (60%)
    - name: "ViT-H-14-quickgelu"
      pretrained: "metaclip_fullcc"
      weight: 0.4         # 이 선생님의 중요도 (40%)
  affinity_temp: 0.1      # Softmax 온도 (낮을수록 sharp)
  w_distill_affinity: 1.0 # Distillation loss 가중치

# === 데이터 설정 ===
data:
  coco_root: "./dataset"  # 데이터셋 경로
  train_augment: true     # 데이터 증강 사용 여부
  max_caps_per_image: 5   # 이미지당 최대 캡션 수
```

### 각 설정값의 의미

| 설정 | 기본값 | 설명 | 조정 가이드 |
|------|--------|------|------------|
| `batch_size` | 128 | 배치 크기 | GPU 메모리에 따라 조정 (OOM시 감소) |
| `learning_rate` | 3e-4 | 학습률 | 학습 불안정시 감소, 느릴 시 증가 |
| `warmup_epochs` | 2 | 워밍업 | 큰 배치에서는 늘리기 |
| `affinity_temp` | 0.1 | 온도 | 낮으면 hard, 높으면 soft |
| `weight` (teacher) | 0.6/0.4 | 선생님 가중치 | 더 좋은 선생님에 더 높은 값 |

---

## 7. 각 스크립트의 함수들

### `train.py` 주요 함수

| 함수 | 역할 | 입력 | 출력 |
|------|------|------|------|
| `train_one_epoch()` | 한 에폭 학습 | DataLoader, Model, Optimizer | Average Loss |
| `evaluate()` | 검증 성능 평가 | DataLoader, Model | Recall@K 메트릭 |
| `main()` | 전체 학습 파이프라인 실행 | Config | 최종 체크포인트 |

```python
# train_one_epoch 의사 코드
def train_one_epoch(model, teacher, dataloader, optimizer):
    for images, tokens, metas in dataloader:
        # 1. Student forward
        student_img_emb, student_txt_emb = model(images, tokens)
        
        # 2. Teacher forward (각자 tokenizer 사용)
        raw_captions = [m['caption'] for m in metas]
        teacher_outputs = teacher(images, raw_captions)
        
        # 3. Loss 계산
        loss = siglip_loss(student_img_emb, student_txt_emb)
        loss += distillation_loss(student_outputs, teacher_outputs)
        
        # 4. Backpropagation
        loss.backward()
        optimizer.step()
```

### `distill.py` 주요 클래스/함수

| 클래스/함수 | 역할 |
|-------------|------|
| `TeacherConfig` | 선생님 모델 설정 데이터클래스 |
| `DistillConfig` | Distillation 전체 설정 |
| `OpenClipTeacher` | open_clip 기반 선생님 (SigLIP, MetaCLIP) |
| `EnsembleTeacher` | 여러 선생님을 묶어서 관리 |
| `create_teacher()` | 설정에 따라 적절한 선생님 생성 |
| `compute_affinity_distill_loss()` | Affinity Distillation 손실 계산 |

```python
# OpenClipTeacher 구조
class OpenClipTeacher:
    def __init__(self, cfg):
        self.model = open_clip.create_model(cfg.name)
        self.tokenizer = open_clip.get_tokenizer(cfg.name)  # 자체 토크나이저!
        
    def forward(self, images, raw_texts):
        # 1. 이미지 인코딩
        img_emb = self.model.encode_image(images)
        
        # 2. 텍스트 토크나이징 (자체 토크나이저 사용)
        tokens = self.tokenizer(raw_texts)
        txt_emb = self.model.encode_text(tokens)
        
        return img_emb, txt_emb
```

### `data.py` 주요 클래스

| 클래스 | 역할 |
|--------|------|
| `CocoCaptionsRetrievalDataset` | COCO 데이터셋 로딩 |
| `JsonlRetrievalDataset` | JSONL 형식 데이터셋 |
| `collate_fn()` | 배치 데이터 정리 |
| `make_datasets()` | 설정에 따라 데이터셋 생성 |

### `loss.py` 주요 함수

| 함수 | 역할 | 논문 출처 |
|------|------|-----------|
| `siglip_loss()` | Sigmoid 기반 대조 학습 | SigLIP (2023) |
| `info_nce_loss()` | Softmax 기반 대조 학습 | CLIP (2021) |
| `text_text_contrastive_loss()` | 텍스트-텍스트 대조 학습 | TULIP (2024) |

### `eval.py` 주요 함수

| 함수 | 역할 |
|------|------|
| `compute_embeddings()` | 전체 데이터의 임베딩 계산 |
| `recall_at_k()` | Recall@K 메트릭 계산 |
| `evaluate_retrieval()` | Image→Text, Text→Image 검색 평가 |

---

## 8. 대회 기준과 적합성

### 🏆 LPCVC 2026 대회 요구사항

| 요구사항 | 우리 프로젝트 | 적합 여부 |
|----------|--------------|-----------|
| 모바일 디바이스 실행 | MobileCLIP2 사용 (~50MB) | ✅ |
| Qualcomm 칩셋 호환 | ONNX 변환 지원 | ✅ |
| Latency < 10ms | 예상 ~5ms (A8 Gen3 기준) | ✅ |
| Retrieval 정확도 | R@10 > 60% 목표 | ⏳ 검증 필요 |
| 오프라인 동작 | 임베딩 기반 검색 | ✅ |

### 📊 평가 메트릭

```
Recall@K: 상위 K개 결과 중 정답이 있는 비율

예시 (Recall@5):
- 검색 결과: [사진A, 사진B, 사진C, 사진D, 사진E]
- 정답: 사진C
- R@5 = 1 (정답이 상위 5개 안에 있으므로)
```

| 메트릭 | 의미 | 대회 목표 (추정) |
|--------|------|-----------------|
| R@1 | 첫 번째가 정답 | > 25% |
| R@5 | 상위 5개 중 정답 | > 50% |
| R@10 | 상위 10개 중 정답 | > 60% |

### ✅ 대회 규칙 준수 체크리스트

- [x] 모델 크기 제한 준수 (< 100MB)
- [x] 외부 API 미사용 (오프라인 동작)
- [x] 표준 프레임워크 사용 (PyTorch → ONNX)
- [x] 재현 가능한 학습 파이프라인
- [ ] Qualcomm AI Hub 프로파일링 완료 (진행 예정)

---

## 9. 사용 시 주의사항

### ⚠️ 환경 관련

| 주의사항 | 설명 | 해결책 |
|----------|------|--------|
| **GPU 메모리 부족** | Dual Teacher는 ~10GB VRAM 필요 | A100 사용 또는 배치 크기 감소 |
| **모델 다운로드 시간** | SigLIP 2 (~7.5GB) 첫 다운로드 오래 걸림 | 사전 다운로드 또는 인내심 |
| **CUDA 버전** | PyTorch와 CUDA 호환성 확인 | CUDA 11.8+ 권장 |

### ⚠️ 학습 관련

| 주의사항 | 설명 | 해결책 |
|----------|------|--------|
| **Overfitting** | 작은 데이터셋에서 과적합 위험 | Data Augmentation 활성화 |
| **Learning Rate 튜닝** | 너무 크면 발산, 너무 작으면 수렴 안 됨 | Grid Search 또는 LR Finder |
| **Warmup 필수** | 워밍업 없이 큰 LR 시작 시 불안정 | warmup_epochs ≥ 2 |
| **EMA 초기화** | EMA 모델은 몇 에폭 후부터 안정화 | 5 에폭 이후 EMA 모델 사용 |

### ⚠️ 토크나이저 관련

| 주의사항 | 설명 | 해결책 |
|----------|------|--------|
| **Teacher-Student 토크나이저 불일치** | 각 모델마다 다른 토크나이저 사용 | 각 Teacher가 자체 토크나이저 사용 (이미 구현됨) |
| **Max Length 초과** | 긴 캡션은 잘림 | max_length=77 기본값 유지 |

### ⚠️ 배포 관련

| 주의사항 | 설명 | 해결책 |
|----------|------|--------|
| **ONNX 변환 호환성** | 일부 연산은 ONNX 미지원 | opset_version=14 이상 사용 |
| **Quantization 정확도 손실** | INT8 변환 시 성능 저하 가능 | Calibration 데이터 충분히 사용 |

---

## 10. 기술 출처 및 참고문헌

### 📚 핵심 논문

| 기술 | 논문 제목 | 저자 | 발표 | 링크 |
|------|-----------|------|------|------|
| **CLIP** | Learning Transferable Visual Models From Natural Language Supervision | Radford et al. (OpenAI) | 2021.01 | [arXiv](https://arxiv.org/abs/2103.00020) |
| **Knowledge Distillation** | Distilling the Knowledge in a Neural Network | Hinton et al. (Google) | 2015.03 | [arXiv](https://arxiv.org/abs/1503.02531) |
| **SigLIP** | Sigmoid Loss for Language Image Pre-Training | Zhai et al. (Google) | 2023.03 | [arXiv](https://arxiv.org/abs/2303.15343) |
| **SigLIP 2** | SigLIP 2: Multilingual Vision-Language Encoders | Google DeepMind | 2025.02 | [arXiv](https://arxiv.org/abs/2502.14786) |
| **MetaCLIP** | Demystifying CLIP Data | Xu et al. (Meta) | 2023.09 | [arXiv](https://arxiv.org/abs/2309.16671) |
| **MetaCLIP 2** | Meta CLIP 2: A Worldwide Scaling Recipe | Meta AI (FAIR) | 2025.07 | [arXiv](https://arxiv.org/abs/2507.xxxxx), NeurIPS 2025 |
| **MobileCLIP** | MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training | Vasu et al. (Apple) | 2024.04 | [arXiv](https://arxiv.org/abs/2311.17049), CVPR 2024 |
| **MobileCLIP2** | MobileCLIP2: Improving Multi-Modal Reinforced Training | Apple | 2025.08 | [TMLR](https://openreview.net/forum?id=mobileclip2) |
| **TinyCLIP** | TinyCLIP: CLIP Distillation via Affinity Mimicking and Weight Inheritance | Wu et al. | 2023.09 | [arXiv](https://arxiv.org/abs/2309.12314), ICCV 2023 |

### 📚 학습 기법 관련

| 기술 | 논문 제목 | 저자 | 발표 | 링크 |
|------|-----------|------|------|------|
| **Mixed Precision** | Mixed Precision Training | Micikevicius et al. (NVIDIA) | 2017.10 | [arXiv](https://arxiv.org/abs/1710.03740) |
| **Cosine Annealing** | SGDR: Stochastic Gradient Descent with Warm Restarts | Loshchilov & Hutter | 2016.08 | [arXiv](https://arxiv.org/abs/1608.03983), ICLR 2017 |
| **Gradient Clipping** | On the difficulty of training Recurrent Neural Networks | Pascanu et al. | 2012.11 | [arXiv](https://arxiv.org/abs/1211.5063) |
| **EMA** | Mean teachers are better role models | Tarvainen & Valpola | 2017.03 | [arXiv](https://arxiv.org/abs/1703.01780) |
| **Label Smoothing** | Rethinking the Inception Architecture for Computer Vision | Szegedy et al. (Google) | 2015.12 | [arXiv](https://arxiv.org/abs/1512.00567) |
| **Linear Warmup** | Accurate, Large Minibatch SGD | Goyal et al. (Facebook) | 2017.06 | [arXiv](https://arxiv.org/abs/1706.02677) |

### 📚 데이터셋

| 데이터셋 | 설명 | 크기 | 링크 |
|----------|------|------|------|
| **MS-COCO Captions** | 이미지-캡션 쌍 데이터셋 | 330K 이미지, 1.5M 캡션 | [Website](https://cocodataset.org/) |
| **Flickr30k** | 이미지-캡션 벤치마크 | 31K 이미지, 155K 캡션 | [Website](http://shannon.cs.illinois.edu/DenotationGraph/) |

### 📚 라이브러리

| 라이브러리 | 버전 | 용도 | 링크 |
|------------|------|------|------|
| PyTorch | ≥2.0 | 딥러닝 프레임워크 | [pytorch.org](https://pytorch.org) |
| open_clip | ≥2.20 | CLIP 변형 모델 로딩 | [GitHub](https://github.com/mlfoundations/open_clip) |
| transformers | ≥4.30 | 토크나이저 | [HuggingFace](https://huggingface.co/transformers) |
| timm | ≥0.9 | Vision 모델 | [GitHub](https://github.com/huggingface/pytorch-image-models) |
| wandb | - | 실험 로깅 | [wandb.ai](https://wandb.ai) |

---

## 🏃 빠른 시작 가이드

### 1️⃣ 설치
```bash
pip install -r requirements.txt
```

### 2️⃣ 학습 시작
```bash
python run_train.py
```

### 3️⃣ 평가
```bash
python scripts/eval.py --checkpoint checkpoints/best.pt
```

### 4️⃣ 배포용 변환
```bash
python scripts/export_onnx_split.py --checkpoint checkpoints/best.pt
```

---

## 🎉 마무리

이 프로젝트는 **"작지만 똑똑한 AI"**를 만드는 것입니다.

```
큰 선생님들 (SigLIP + MetaCLIP)
        ↓
    [열심히 가르침]
        ↓
작은 학생 (MobileCLIP2)
        ↓
    [스마트폰에서 동작!] 📱
```

---

*문서 최종 업데이트: 2026년 2월*
*질문이 있으시면 언제든 물어보세요! 🙋‍♂️*
