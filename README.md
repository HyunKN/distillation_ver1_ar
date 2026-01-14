# LPCVC 2026 Track 1 – CLIP Lite (FastViT + TinyText) Image↔Text Retrieval

> Snapdragon(NPU/HTP) 환경에서 동작할 **초경량 Image↔Text Retrieval(검색)** 모델을 학습하고,  
> ONNX로 export하여 Qualcomm AI Hub에서 성능(지연/속도)을 확인하는 프로젝트입니다.

* 해당 레포는 초기 프로젝트 시 내용 공유를 위해 임시로 작성해본 프로젝트입니다.

---

## 1) 우리가 만드는 것

```
입력(이미지 또는 텍스트) → 임베딩 벡터 생성 → 유사도 기반 검색
```

**목표:**
- **정확도**: Recall@K (K=1/5/10)
- **추론 성능**: 모바일 NPU에서 빠르게 동작
- **대회 입력/출력 규격 준수** (텍스트 길이 77 고정 등)

---

## 2) 프로젝트 구조

```
LPCV_Track1_ImgToText/
│
│    # 데이터셋은 따로 다운로드가 필요합니다. (용량이 너무 큼)
├── dataset/coco/
│   ├── train2017/                     # *.jpg
│   ├── val2017/                       # *.jpg
│   └── annotations/
│       ├── captions_train2017.json
│       └── captions_val2017.json
├── config.yaml                  # 전체 설정(모델/데이터/학습/증류/추출)
├── scripts/
│   ├── train.py                 # 학습 실행 엔트리포인트
│   ├── eval.py                  # 평가 실행
│   ├── export_onnx.py           # ONNX export (통합 모델)
│   └── export_onnx_split.py     # ONNX export (분리, 대회 샘플 호환)
│
└── src/lpcvc_retrieval/
    ├── model.py                 # Student CLIP (FastViT + TinyText)
    ├── train.py                 # 학습 루프(스케줄러/클리핑/로그/체크포인트)
    ├── distill.py               # Teacher 준비 + affinity distill 계산
    ├── data.py                  # COCO/JSONL 로더 + augmentation + collate
    ├── losses.py                # contrastive / ranking / label smoothing
    └── metrics.py               # Recall@K + 양방향(I2T/T2I/Mean)
```

---

## 3) 사용한 기술 (현재 코드 기준)

### 모델 (Architecture)
- **Vision Encoder**: FastViT-S12 (`timm`, `fastvit_s12.apple_in1k`)
- **Text Encoder**: 경량 Tiny Transformer (직접 구현)
  - 입력: CLIP tokenizer 기반 token id
  - 길이: `context_length = 77` 고정
- image/text 각각 projection → **L2 normalize**
- learnable `logit_scale`(temperature) 사용 + 범위 clamp

### 학습 안정화 (Training loop)
- **AMP** (mixed precision): 학습 속도/VRAM 최적화
- **Warmup + Cosine LR scheduler**
- **Gradient clipping**
- **Logit scale clamping**: temperature 폭주 방지 (기본 `[-4.6, 4.6]`)

### Loss
- **CLIP symmetric contrastive loss** (기본): image↔text 정렬 (InfoNCE)
- **Hard negative 강화** (옵션): pairwise ranking loss
- **Label smoothing** (옵션): 과적합 완화

### 데이터 (Generalization)
- **Train augmentation**: RandomResizedCrop / HorizontalFlip / ColorJitter
- COCO는 이미지당 캡션이 여러 개라 `max_captions_per_image`로 학습 다양화

### 평가
- **Recall@1/5/10**
- **양방향 평가**: I2T(Image→Text), T2I(Text→Image), Mean(평균)

### 제출/배포 (LPCVC 대회 샘플 규격 준수)
- **분리 ONNX export** (opset 18, 대회 샘플 호환)
  - `image_encoder.onnx`: input `image` float32 (1,3,224,224) → output `embedding`
  - `text_encoder.onnx`: input `text` int32 (1,77) → output `text_embedding`
- Teacher는 학습에만 사용되며 ONNX에는 포함되지 않음
- 이미지 정규화(CLIP mean/std)는 `normalize_input=true`로 모델 내부에서 처리
- 텍스트는 평가 규격(int32) 입력 후 모델 내부에서 int64로 캐스팅 (embedding 안정성)

---

## 4) 데이터셋 준비 (COCO)

`config.yaml`에서 아래처럼 설정되어 있다고 가정합니다:

```yaml
data:
  mode: coco
  coco_root: dataset/coco
  train_split: train2017
  val_split: val2017
  train_captions_json: annotations/captions_train2017.json
  val_captions_json: annotations/captions_val2017.json
```

폴더 구조:
```
dataset/coco/
├── train2017/                     # *.jpg
├── val2017/                       # *.jpg
└── annotations/
    ├── captions_train2017.json
    └── captions_val2017.json
```
데이터셋: https://cocodataset.org/#download

**사용된 데이터셋 규격:**
- **2017 Train images** [118K/18GB]
- **2017 Val images** [5K/1GB]
- **2017 Train/Val annotations** [241MB]
---

## 5) 설치 (Windows CMD 기준)

### 5.1 가상환경 생성/활성화
```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 5.2 설치

**pyproject.toml 기반:**
```cmd
pip install -U pip
pip install -e .
```

**requirements.txt 기반:**
```cmd
pip install -r requirements.txt
```

### 5.3 GPU 확인
```cmd
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 6) 실행 순서 (Train → Eval → Export)

### 6.1 학습 시작
```cmd
python scripts\train.py --config config.yaml
```

학습 결과 체크포인트:
- `runs/lpcvc_clip_lite/best.pt`
- `runs/lpcvc_clip_lite/last.pt`
- `runs/lpcvc_clip_lite/epoch_N.pt` (epoch별)

### 6.2 평가
```cmd
python scripts\eval.py --config config.yaml --ckpt runs\lpcvc_clip_lite\best.pt
```

### 6.3 ONNX export

**방법 1: 통합 모델 (단일 ONNX)**
```cmd
python scripts\export_onnx.py --config config.yaml --ckpt runs\lpcvc_clip_lite\best.pt --out model.onnx
```

**방법 2: 분리 모델 (대회 샘플 호환, 권장)**
```cmd
python scripts\export_onnx_split.py --config config.yaml --ckpt runs\lpcvc_clip_lite\best.pt --out_dir exported_onnx
```

출력:
- `exported_onnx/image_encoder.onnx`
- `exported_onnx/text_encoder.onnx`

| 항목 | Image Encoder | Text Encoder |
|------|---------------|---------------|
| Input Name | `image` | `text` |
| Input Shape | `(1,3,224,224)` float32 | `(1,77)` int32 |
| Output Name | `embedding` | `text_embedding` |
| Opset | 18 | 18 |

---

## 7) Recall@K (Recall@1/5/10) 란?

### Recall(재현율)이란?
Retrieval에서는 **"정답을 얼마나 잘 찾아냈는가"**를 봅니다.

쿼리(query) 1개를 주고 모델이 후보들을 점수로 정렬했을 때,  
**정답이 상위 K개 안에 들어가면 성공**입니다.

```
Recall@K = (성공한 쿼리 수 / 전체 쿼리 수) × 100%
```

### K=1/5/10의 의미
| 지표 | 의미 |
|------|------|
| **Recall@1** | 정답이 1등(top-1)에 있는 비율 |
| **Recall@5** | 정답이 상위 5개 안에 있는 비율 |
| **Recall@10** | 정답이 상위 10개 안에 있는 비율 |

### I2T / T2I / Mean
| 지표 | 의미 |
|------|------|
| **I2T** | 이미지로 정답 캡션을 찾는 성능 |
| **T2I** | 텍스트로 정답 이미지를 찾는 성능 |
| **Mean** | I2T와 T2I의 평균 |

---

## 7-1) eval.py 평가 방식 (채점 로직)

### 평가 흐름

```
1. 체크포인트 로드 → 2. 전체 val 데이터 임베딩 추출 → 3. 유사도 계산 → 4. Recall@K 산출
```

### Step 1: 모든 이미지/텍스트 임베딩 추출
```python
for imgs, toks, _ in loader:
    img_emb, txt_emb = model(imgs, toks)  # [B, 256]
    all_img.append(img_emb)
    all_txt.append(txt_emb)

image_emb = torch.cat(all_img, dim=0)  # [N, 256] 전체 이미지
text_emb = torch.cat(all_txt, dim=0)   # [N, 256] 전체 텍스트
```

### Step 2: 유사도 행렬(Similarity Matrix) 계산
```python
# I2T: 각 이미지가 모든 텍스트와 얼마나 유사한지
sim_i2t = image_emb @ text_emb.t()  # [N, N] 코사인 유사도

# T2I: 각 텍스트가 모든 이미지와 얼마나 유사한지
sim_t2i = text_emb @ image_emb.t()  # [N, N]
```

### Step 3: 순위(Rank) 계산
```python
ranks = torch.argsort(sim, dim=1, descending=True)  # 유사도 높은 순 정렬
# 정답은 대각선 위치 (i번째 이미지의 정답은 i번째 텍스트)
```

### Step 4: Recall@K 계산
```python
# 정답이 상위 K개 안에 있으면 성공
for k in [1, 5, 10]:
    recall_at_k = (정답_순위 < k).float().mean()
```

### 출력 예시
```
I2T: R@1=25.30% R@5=48.20% R@10=60.50%
T2I: R@1=22.10% R@5=45.80% R@10=58.30%
Mean: R@1=23.70% R@5=47.00% R@10=59.40%
```

### 핵심 가정
> **i번째 이미지와 i번째 텍스트가 정답 쌍**이라고 가정 (COCO Captions 구조)

---

## 8) Distillation (Teacher → Student 지식 증류)

### Distillation이란?
**선생(Teacher)** 모델이 이미 학습해둔 지식을 이용해서  
**학생(Student)** 모델이 더 잘/더 빨리 학습하도록 추가 신호를 주는 것입니다.

### 우리가 사용하는 방식: TinyCLIP의 Affinity Mimicking

> **TinyCLIP 논문**에서 제안된 방식을 CLIP에 맞게 구현했습니다.

1. Teacher/Student 각각 image/text 임베딩을 생성
2. 배치 내부 **유사도 행렬(similarity matrix)** 생성
3. Student의 유사도 분포가 Teacher와 비슷해지도록 **KL Divergence**로 학습
4. (옵션) **Selective distillation**: Student가 자신 없을 때만 distill 적용

```python
# distill.py Line 120
"""
Computes Tiny-CLIP style affinity mimicking loss.
- We compare similarity distributions within a batch, not raw embeddings dims.
- Optional selective distillation: only rows with low student confidence are distilled.
"""
```

### Distill을 켜려면 (config.yaml)

**1. Teacher 사용 ON:**
```yaml
distill:
  use_teacher: true
  teacher_model_name: ViT-B-32
  teacher_pretrained: openai
```

**2. Distill loss 가중치 ON (0보다 크게):**
```yaml
loss:
  w_distill_affinity: 0.1   # 0.1~0.3부터 시작 추천 (0이면 OFF)
```

### Distill 옵션들
```yaml
distill:
  distill_margin_thr: 0.2     # selective distill: 자신 없을 때만 적용
  affinity_temp: 0.1          # affinity softmax temperature
  affinity_columns: false     # true면 T2I 방향(열)도 같이 맞춤 (비용↑)
```

### Distill OFF
```yaml
distill:
  use_teacher: false
loss:
  w_distill_affinity: 0.0
```

> **참고**: distill은 학습 때만 사용되며, ONNX export/추론 모델에는 teacher가 포함되지 않습니다.

---

## 9) config.yaml로 기능 ON/OFF 하는 규칙

| 타입 | ON | OFF |
|------|-----|-----|
| boolean | `true` | `false` |
| 가중치 (`w_*`) | `> 0` | `0` |

```yaml
loss:
  w_rank: 0.1           # ranking ON, 0이면 OFF
  label_smoothing: 0.1  # smoothing ON, 0이면 OFF
data:
  train_augment: true   # augmentation ON
```

---

## 10) 실험 추천 순서 (안전한 튜닝 루트)

| Step | 설정 |
|------|------|
| **1. Baseline** (증류 OFF) | `use_teacher: false`, `w_distill_affinity: 0.0` |
| **2. Rank loss 추가** | `w_rank: 0.1` |
| **3. Label smoothing + 캡션 다양화** | `label_smoothing: 0.1`, `max_captions_per_image: 5` |
| **4. Distill ON** | `use_teacher: true`, `w_distill_affinity: 0.1~0.3` |

> ⚠️ Distill 가중치가 너무 크면 초반 학습이 무너질 수 있음

---

## 11) 주의사항 / 트러블슈팅

### (A) lr이 0으로 찍히면 학습이 망가질 수 있음
- tqdm 로그에 `lr=0.00e+00`이 지속되면 학습이 거의 진행되지 않음
- Warmup/Scheduler 구현 문제 의심

### (B) scheduler.step() 경고
- 일반적으로 `optimizer.step()` → `scheduler.step()` 순서
- AMP/GradScaler에서 overflow로 optimizer step이 스킵되는 경우도 있음

### (C) ONNX export 경고 (advanced indexing)
- 특정 indexing이 ONNX에서 여러 op로 풀리며 경고 발생
- 입력에 음수 인덱싱이 발생하지 않는지 확인 필요

---

## 12) Qualcomm AI Hub 업로드 및 프로파일링

### 사전 준비
1. [Qualcomm AI Hub](https://aihub.qualcomm.com/) 계정 생성
2. `qai_hub` 패키지 설치:
```cmd
pip install qai-hub
```
3. API 키 설정:
```cmd
qai-hub configure --api_token YOUR_API_TOKEN
```

### Step 1: 분리 ONNX Export (타임스탬프 자동)

```cmd
python scripts\export_onnx_split.py --ckpt runs\lpcvc_clip_lite\best.pt --prefix lpcvc
```

출력 예시:
```
exported_onnx/이미지인코더.onnx
exported_onnx/텍스트인코더.onnx
```

> 💡 `--prefix` 옵션으로 모델명 구분 가능. 타임스탬프로 덮어쓰기 방지됨.

### Step 2: AI Hub 컴파일 & 프로파일링

```cmd
python compile_and_profile.py --img_name "이미지파일명" --txt_name "텍스트파일명"
```
테스트 중에는 onnx 파일명은 
lpcvc_fastvitS12_e3_best_image 처럼 세부 config 사항을 명시하는 것을 추천합니다.
프로젝트명/모델명/에폭/상태/인코더종류
(업로드 후 모델명 수정 추천)

**옵션:**
| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--onnx_dir` | `exported_onnx` | ONNX 파일 디렉토리 |
| `--img_name` | `image_encoder.onnx` | 이미지 인코더 파일명 |
| `--txt_name` | `text_encoder.onnx` | 텍스트 인코더 파일명 |
| `--device` | `Snapdragon 8 Elite QRD` | 타겟 디바이스 |
| `--skip_profile` | - | 프로파일링 스킵 |

### 실행 흐름

```
1. ONNX 파일 로드 & 유효성 검사
2. QAI Hub에 컴파일 job 제출 (image/text 각각)
3. 컴파일 완료 대기 (wait)
4. 컴파일된 모델로 프로파일 job 제출
5. Job ID 출력 → AI Hub 웹에서 결과 확인
```

### 결과 확인
- 웹: https://aihub.qualcomm.com/jobs
- 프로파일 결과: 추론 시간, 레이어별 지연, 메모리 사용량 등

### I/O 규격 (대회 샘플 준수)

| 모델 | Input Name | Input Shape | Input Type | Output Name |
|------|------------|-------------|------------|-------------|
| Image Encoder | `image` | `(1,3,224,224)` | float32 | `embedding` |
| Text Encoder | `text` | `(1,77)` | int32 | `text_embedding` |

---

## 13) References (관련 논문)

| 논문 | 링크 |
|------|------|
| **CLIP** (Contrastive Language–Image Pre-training) | https://arxiv.org/abs/2103.00020 |
| **FastViT** | https://arxiv.org/abs/2303.14189 |
| **MobileCLIP** | https://arxiv.org/abs/2311.17049 |
| **TinyCLIP** (Affinity Mimicking) | https://arxiv.org/abs/2309.12314 |

---

## License

This project is for LPCVC 2026 competition purposes.
