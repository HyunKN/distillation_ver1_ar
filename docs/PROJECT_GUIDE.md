# MobileCLIP2 Retrieval Optimization — 프로젝트 가이드

> 최종 갱신: 2026-03-09  
> 이 문서는 현재 저장소 코드를 기준으로 프로젝트의 목적, 구조, 동작 원리, 모든 모듈의 상세 설명을 기록합니다.

---

## 1. 프로젝트 개요

### 1.1 목적

LPCVC 2026 Track 1 — 모바일 환경에서 이미지-텍스트 검색(Image-Text Retrieval)을 수행하는 경량 모델을 학습합니다.

**핵심 전략**: Apple의 사전학습된 `MobileCLIP2-S0`(학생)을 두 개의 대형 teacher 모델(`SigLIP 2 Giant`, `PE-Core bigG-14-448`)로부터 지식 증류(Knowledge Distillation)하여, 모바일 디바이스에서도 높은 retrieval 성능을 달성합니다.

### 1.2 대회 정보

- **대회**: [LPCVC 2026 Track 1](https://lpcv.ai/) — Low Power Computer Vision Challenge
- **목표 디바이스**: Qualcomm XR2 Gen 2 (Proxy)
- **평가 지표**: I2T / T2I Recall@K (K=1, 5, 10)
- **배포 형식**: ONNX → QNN (Qualcomm AI Hub compile)

### 1.3 핵심 설계 결정

| 결정 | 근거 |
|------|------|
| MobileCLIP2-S0 학생 | 모바일 최적화된 사전학습 멀티모달 모델 (Apple, MIT) |
| SigLIP 2 + PE-Core 듀얼 teacher | 서로 다른 강점(정렬 vs 강건성)을 보완 |
| Adaptive teacher weighting (기본값) | 고정 비율보다 데이터 적응적 mixing이 안전 |
| Offline feature 모드 지원 | teacher VRAM 0으로 반복 실험 가능 |
| SigLIP loss (Sigmoid 기반) | 작은 배치에서 Softmax보다 안정적 |
| ONNX 분리 export | image/text encoder 독립 배포 |

---

## 2. 전체 실행 흐름

```text
1. 데이터 전처리       scripts/preprocess/parse_lpcvc_sources.py
                       → train.jsonl, val.jsonl 생성
                            │
2. (선택) Feature 추출  scripts/extract_features.py
                       → features/teacher_*_{train,val}.pt
                            │
3. 학습                 run_train.py → src/lpcvc_retrieval/train.py
                       → runs/lpcvc_clip_lite/{best,last,epoch_*}.pt
                            │
4. 평가                 scripts/eval.py
                       → I2T/T2I Recall@1,5,10 출력
                            │
5. ONNX Export          scripts/export_onnx_split.py
                       → exported_onnx/{image_encoder,text_encoder}.onnx
                            │
6. QAI Hub 컴파일       compile_and_profile.py
                       → Qualcomm QNN DLC 컴파일 및 프로파일
```

---

## 3. 모델 아키텍처

### 3.1 학생 모델 — MobileCLIP2-S0

| 항목 | 값 |
|------|-----|
| 모델 | MobileCLIP2-S0 |
| 출처 | [Apple ML Research](https://github.com/apple/ml-mobileclip) |
| HuggingFace | [apple/MobileCLIP2-S0](https://huggingface.co/apple/MobileCLIP2-S0) |
| 코드 라이선스 | MIT |
| 가중치 라이선스 | Apple model card terms |
| 사전학습 | `dfndr2b` (DataFilterNetworks Balanced) |
| 이미지 입력 | `(B, 3, 224, 224)` float32, [0, 1] 범위 |
| 텍스트 입력 | `(B, 77)` int32 token IDs |
| 기본 임베딩 차원 | 256 |
| 출력 | L2 normalized 임베딩 |

**Variant 지원**: S0, S2, S3, S4, B, L-14 (현재 기본값: S0)

학생 모델이 원본 출력 차원과 설정 `embed_dim`이 다르면 **projection layer** (`nn.Linear`)가 자동 추가됩니다.

### 3.2 Teacher 모델

#### Teacher 1 — SigLIP 2 Giant

| 항목 | 값 |
|------|-----|
| 모델 | ViT-gopt-16-SigLIP2-256 |
| 출처 | [Google Research](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md) |
| HuggingFace | [timm/ViT-gopt-16-SigLIP2-256](https://huggingface.co/timm/ViT-gopt-16-SigLIP2-256) |
| 라이선스 | Apache-2.0 |
| 입력 크기 | 256 x 256 |
| pretrained | `webli` |
| 강점 | 이미지-텍스트 정렬 품질 |

#### Teacher 2 — PE-Core bigG-14-448

| 항목 | 값 |
|------|-----|
| 모델 | PE-Core-bigG-14-448 |
| 출처 | [Meta FAIR](https://github.com/facebookresearch/perception_models) |
| HuggingFace | [facebook/PE-Core-G14-448](https://huggingface.co/facebook/PE-Core-G14-448) |
| 라이선스 | Apache-2.0 |
| 입력 크기 | 448 x 448 |
| pretrained | `meta` |
| 강점 | 시각적 강건성, perception prior |

#### Teacher 공통 특징

- 모두 `open_clip.create_model_and_transforms()`로 로드 → [OpenCLIP](https://github.com/mlfoundations/open_clip) (MIT)
- 각 teacher는 **자체 tokenizer** 사용 (`open_clip.get_tokenizer()`)
- Teacher 입력 크기에 맞게 **bicubic interpolation**으로 resize
- Teacher 전용 **mean/std**로 normalize (preprocess에서 자동 추출)
- 학습 중 teacher는 `eval()` 모드, `@torch.no_grad()`로 실행

---

## 4. 데이터 파이프라인

> 코드: `src/lpcvc_retrieval/data.py`

### 4.1 데이터 모드

| 모드 | 설명 | Config 키 |
|------|------|-----------|
| `jsonl` (기본) | JSONL 파일 기반 | `data.train_jsonl`, `data.val_jsonl` |
| `coco` | COCO annotations JSON 직접 로드 | `data.coco_root`, `data.train_captions_json` |

### 4.2 JSONL 계약

```json
{"image":"coco/train2017/000000000009.jpg","captions":["a cat on sofa"],"source":"coco"}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `image` | string | 필수 | `image_root`에 대한 상대 경로 |
| `captions` | list[string] | 필수 | 캡션 목록 (1개 이상) |
| `source` | string | 선택 | 데이터셋 출처 (`coco`, `flickr30k`, `open_images`, `wit`) |

### 4.3 토크나이저

```python
CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
```

- 출처: [HuggingFace transformers](https://huggingface.co/openai/clip-vit-base-patch32)
- `max_length=77`, `padding="max_length"`, `truncation=True`
- 출력은 `int32`로 변환

### 4.4 데이터 증강

`_img_transform_train(augment=True)`가 적용하는 증강:

| 증강 | 파라미터 | 효과 |
|------|----------|------|
| `RandomResizedCrop` | scale=(0.8, 1.0), 224px | 부분 물체 인식 강제 |
| `RandomHorizontalFlip` | p=0.5 | 실효 데이터셋 2배 |
| `ColorJitter` | brightness=0.2, contrast=0.2, saturation=0.1 | 조명 강건성 |

평가 시에는 `Resize(224) → CenterCrop(224) → ToTensor()` 만 적용합니다.

### 4.5 데이터셋 클래스

#### `JsonlRetrievalDataset`

| 동작 | 학습 (is_train=True) | 평가 (is_train=False) |
|------|---------------------|----------------------|
| 샘플 단위 | 이미지 단위 (캡션 목록 보관) | 캡션 단위 (전개) |
| 캡션 선택 | 랜덤 1개 샘플링 | 전체 사용 |
| 튜플 형태 | `(img_rel, image_id, caps, source)` | `(img_rel, image_id, caption, ann_id, source)` |

반환 메타 딕셔너리: `image_id`, `ann_id`, `caption`, `caption_idx`, `img_rel`, `source`

**`set_forced_caption_indices()`**: offline feature 모드에서 feature 추출 시와 동일한 캡션을 사용하도록 강제 주입합니다.

#### `CocoCaptionsRetrievalDataset`

COCO captions JSON을 직접 로드합니다. `image_id`를 유지하여 COCO-style 평가를 가능하게 합니다. train/eval 모두 `source="coco"` 메타를 포함합니다.

#### `OfflineFeatureDataset`

미리 추출한 teacher 임베딩을 base dataset과 함께 반환하는 래퍼입니다.

**검증 항목** (6가지):
1. 데이터셋 길이와 feature 파일 길이 일치
2. `img_embs`, `txt_embs` shape 일치
3. `sample_count` 일치
4. Teacher 간 `dataset_fingerprint` 일치
5. 현재 데이터셋과 feature 파일의 `dataset_fingerprint` 일치
6. Teacher 간 `caption_indices` 일치

Train split에서는 `caption_indices`를 base dataset에 재주입 → feature 추출 시와 학습 시 캡션 불일치 방지

#### `collate_fn`

배치를 `(imgs, toks, metas)` 또는 `(imgs, toks, metas, teacher_batch)`로 조합합니다. Offline 모드에서는 teacher 임베딩이 함께 반환됩니다.

#### `make_datasets(cfg, tokenizer)`

Config 기반으로 적절한 데이터셋 클래스를 생성하며, `distill.offline_feature_dir`이 존재하면 자동으로 `OfflineFeatureDataset`으로 래핑합니다.

### 4.6 전처리 스크립트

#### `scripts/preprocess/parse_lpcvc_sources.py` (503줄)

4개 데이터셋(COCO, Flickr30k, Open Images, WIT)을 하나의 JSONL로 통합합니다.

주요 함수:

| 함수 | 역할 |
|------|------|
| `_iter_coco()` | COCO annotations JSON에서 이미지-캡션 쌍 추출 |
| `_iter_flickr30k()` | Flickr30k `results.csv`에서 추출 |
| `_iter_open_images()` | Open Images `localized_narratives_*.jsonl`에서 추출 |
| `_iter_wit()` | WIT `.tsv.gz`에서 추출 (다국어 포함) |
| `_dedupe_records()` | 이미지 경로 기준 중복 제거 |
| `_split_train_val()` | 이미지 단위 train/val 분리 |
| `_parse_source_caps()` | source별 이미지 수 상한 설정 |

CLI 인자:
- `--data_root`: 원본 이미지 루트
- `--out_dir`: JSONL 출력 경로
- `--val_ratio`: 검증셋 비율 (기본 0.01)
- `--source_caps`: source별 이미지 수 제한 (예: `open_images=40000,coco=0`)

#### `scripts/preprocess/materialize_upload_subset.py` (175줄)

JSONL에서 참조하는 이미지만 모아 업로드용 서브셋을 생성합니다. `--dry_run`으로 용량 추정만 가능합니다.

---

## 5. 설정 시스템

> 코드: `src/lpcvc_retrieval/config.py` (69줄)

### 5.1 CfgNode

YAML 딕셔너리를 dot-access로 사용할 수 있게 래핑합니다.

```python
cfg = load_config("config.yaml")
cfg.model.embed_dim  # 256
cfg.data.mode        # "jsonl"
```

### 5.2 `load_config(path, overrides)`

1. `yaml.safe_load()`로 YAML 로드
2. `--override key=value` 목록을 `_deep_set()`으로 반영 (dotted key 지원)
3. 값은 `_parse_value()`로 YAML 파싱 (숫자, bool, 리스트 등 자동 변환)

### 5.3 `resolve_device(device)`

`"auto"` → CUDA 사용 가능하면 `"cuda"`, 아니면 `"cpu"` 반환

### 5.4 Configuration Reference

```yaml
data:
  mode: jsonl                          # jsonl | coco
  batch_size: 128
  image_root: dataset
  train_jsonl: dataset/prepared_jsonl/train.jsonl
  val_jsonl: dataset/prepared_jsonl/val.jsonl
  max_captions_per_image: 5            # 학습 시 캡션 샘플링 풀 크기
  num_workers: 4
  train_augment: true

model:
  mobileclip2_variant: S0              # S0 | S2 | S3 | S4 | B | L-14
  embed_dim: 256                       # 최종 임베딩 벡터 차원
  checkpoint_path: null                # null이면 auto-download
  freeze_backbone: false               # backbone 동결 여부
  # temperature_init: 0.07             # config에 존재하지만 코드에서 미사용
  # 실제 logit_scale 초기값은 mobileclip2.py에서 log(1/0.07)로 하드코딩
  logit_scale_min: -4.6
  logit_scale_max: 4.6

distill:
  use_teacher: true
  teachers:
    - name: ViT-gopt-16-SigLIP2-256
      pretrained: webli
      input_size: 256
    - name: PE-Core-bigG-14-448
      pretrained: meta
      input_size: 448
  static_teacher_weights: [0.5, 0.5]   # static 모드 전용 mixing 비율
  adaptive_teacher_weight: true         # true이면 adaptive/adaptive_source 자동 승격
  teacher_weight_mode: adaptive         # static | adaptive | adaptive_source
  adaptive_teacher_tau: 0.07            # softmax temperature
  adaptive_teacher_w_min: 0.20          # teacher collapse 방지 최소 가중치
  source_teacher_weights: {}            # adaptive_source 전용 source prior
  distill_margin_thr: 0.2              # selective distill threshold
  affinity_temp_start: 0.12            # epoch 1 temperature (softer)
  affinity_temp_end: 0.07              # final epoch temperature (sharper)
  affinity_temp_schedule: cosine        # constant | linear | cosine
  affinity_columns: false              # column-wise distillation 추가 여부
  offline_feature_dir: null             # offline feature 경로

loss:
  w_contrastive: 1.0                   # SigLIP loss 비중
  w_rank: 0.1                          # Pairwise ranking loss
  w_hard_negative: 0.1                 # Hard negative contrastive (BLIP/FG-CLIP)
  w_text_text: 0.05                    # Text-text contrastive (TULIP)
  w_distill_affinity: 0.8             # Affinity distillation
  rank_k: 3
  rank_margin: 0.1
  hard_negative_k: 5
  label_smoothing: 0.0

train:
  epochs: 10
  lr: 5.0e-4
  weight_decay: 0.05
  warmup_epochs: 3.0
  grad_clip: 1.0
  amp: true
  use_compile: true                    # torch.compile (PyTorch 2.x)
  use_ema: true                        # EMA 사용
  ema_decay: 0.999
  eval_every_epochs: 1
  log_every: 20
  use_wandb: false
  wandb_project: lpcvc-clip-lite

output:
  out_dir: runs/lpcvc_clip_lite

export:
  onnx_path: model.onnx
  opset: 18

seed: 42
device: cuda
```

---

## 6. 학생 모델 상세

> 코드: `src/lpcvc_retrieval/mobileclip2.py`, `src/lpcvc_retrieval/model.py`

### 6.1 MobileCLIP2Student 클래스

| 메서드 | 시그니처 | 동작 |
|--------|----------|------|
| `__init__` | `(variant, embed_dim, freeze_backbone, checkpoint_path)` | 모델 로드 + projection 설정 |
| `_load_model` | `()` | `open_clip` 로드 → reparameterize → projection 추가 |
| `encode_image` | `(images) → [B, D]` | 이미지 인코딩 + projection + L2 normalize |
| `encode_text` | `(input_ids) → [B, D]` | 텍스트 인코딩 + projection + L2 normalize |
| `forward` | `(images, text_input) → (img_emb, txt_emb)` | 양쪽 동시 인코딩 |
| `get_tokenizer` | `() → tokenizer` | open_clip 토크나이저 반환 |

**학습 가능 파라미터**:
- `logit_scale`: `nn.Parameter(log(1/0.07))` — 대조 학습 temperature
- `logit_bias`: `nn.Parameter(0.0)` — SigLIP loss에서 사용하는 bias

**reparameterize**: `mobileclip.modules.common.mobileone.reparameterize_model()` — MobileCLIP2는 학습 시 multi-branch 구조를 사용하고, 추론 시 단일 branch로 합칩니다.

### 6.2 create_model_from_config

```python
def create_model_from_config(cfg, vocab_size=None, eos_id=None) -> MobileCLIP2Student
```

Config에서 `mobileclip2_variant`, `embed_dim`, `freeze_backbone`, `checkpoint_path`를 읽어 모델을 생성합니다. `vocab_size`와 `eos_id`는 하위 호환을 위해 파라미터로 남아 있지만 사용되지 않습니다.

### 6.3 OnnxWrapper

ONNX export를 위한 래퍼입니다. `(image, text_input) → (img_emb, txt_emb)` 인터페이스를 제공합니다.

---

## 7. Teacher 및 Distillation

> 코드: `src/lpcvc_retrieval/distill.py`

### 7.1 설정 스키마

#### `TeacherConfig`

```python
@dataclass
class TeacherConfig:
    name: str              # open_clip 모델 이름
    pretrained: str        # pretrained 키워드
    input_size: int = None # 강제 입력 크기 (없으면 모델에서 추론)
```

`teachers` 목록의 각 항목이고, teacher 자체 정의 (이름, pretrained, 입력 크기)만 포함합니다.

#### `DistillConfig`

핵심 필드:

| 필드 | 타입 | 기본값 | 역할 |
|------|------|--------|------|
| `use_teacher` | bool | False | teacher 사용 여부 |
| `teachers` | list[TeacherConfig] | [] | teacher 목록 |
| `static_teacher_weights` | list[float] or None | None | `static` 모드 전용 mixing 비율 |
| `adaptive_teacher_weight` | bool | False | adaptive 활성화 |
| `teacher_weight_mode` | str | "static" | `static` / `adaptive` / `adaptive_source` |
| `adaptive_teacher_tau` | float | 0.07 | adaptive softmax temperature |
| `adaptive_teacher_w_min` | float | 0.0 | teacher collapse 방지 최소 가중치 |
| `source_teacher_weights` | dict | {} | source별 teacher prior |
| `offline_feature_dir` | str or None | None | offline feature 경로 |

**`__post_init__` 로직**:
1. Legacy `teacher_model_name` 형식 → `teachers` 리스트로 변환
2. YAML dict를 `TeacherConfig` 객체로 정규화
3. YAML에 `weight` 키가 있으면 `static_teacher_weights`로 자동 마이그레이션
4. `static_teacher_weights` 길이와 `teachers` 길이 불일치 시 `ValueError`

### 7.2 Teacher 모듈

#### `OpenClipTeacher`

| 동작 | 설명 |
|------|------|
| 로드 | `open_clip.create_model_and_transforms()` |
| 토크나이저 | `open_clip.get_tokenizer()` (teacher별 독립) |
| Normalize | teacher 전용 mean/std (preprocess에서 추출, 없으면 OpenAI 기본값) |
| Resize | 학생 입력과 teacher 입력 크기가 다르면 bicubic interpolation |
| 출력 | L2 normalized `(img_features, txt_features)` |

#### `EnsembleTeacher`

`nn.ModuleList`로 여러 `OpenClipTeacher`를 보유하고, `forward()`에서 각 teacher의 결과를 리스트로 반환합니다.

#### `create_teacher(cfg, device)`

Teacher 수에 따라 `OpenClipTeacher` 또는 `EnsembleTeacher`를 반환합니다. Teacher가 0개이면 `ValueError`를 발생시킵니다.

### 7.3 Teacher Routing

#### 세 가지 모드

| 모드 | Prior | Adaptive score | 설명 |
|------|-------|---------------|------|
| `static` | `static_teacher_weights` | 미사용 | 고정 비율 mixing |
| `adaptive` | 균등 (1/N) | 사용 | 배치/샘플 기반 동적 mixing (기본값) |
| `adaptive_source` | `source_teacher_weights`의 source prior | 사용 | source별 prior + adaptive score |

`adaptive_teacher_weight=true`이고 `teacher_weight_mode=static`이면, `source_teacher_weights`가 있으면 `adaptive_source`로, 없으면 `adaptive`로 **자동 승격**됩니다.

#### Adaptive score 계산 (`_teacher_quality_margin_per_row`)

```
score_i = mean(positive similarities) - max(negative similarity)
```

1. Teacher별 similarity matrix 계산 `t_sim = t_img @ t_txt.T`
2. `image_id` 기반으로 positive/negative mask 생성
3. 각 샘플에 대해 positive mean - hardest negative = margin
4. 이 margin이 큰 teacher가 해당 샘플에 대해 더 좋은 teacher

#### Weight 계산 흐름

```text
1. _build_prior_weight_matrix()
   ├─ static  → static_teacher_weights
   ├─ adaptive → uniform (1/N)
   └─ adaptive_source → source별 prior (없으면 uniform)

2. _adaptive_weights_from_scores()
   └─ teacher quality scores를 softmax(tau)로 변환
      └─ w_min으로 floor 적용 (collapse 방지)

3. _combine_teacher_weights()
   └─ prior * adaptive → normalize
```

#### Source prior 해석 (`_resolve_source_prior_weights`)

`source_teacher_weights`에서 현재 샘플의 source key를 찾아 teacher별 prior weight를 반환합니다. 해당 source가 없으면 `None` → uniform prior로 대체됩니다.

### 7.4 Affinity Distillation

핵심 함수: `compute_affinity_distill_loss()`

**Single Teacher 경로**:
1. `s_sim = student_img @ student_txt.T`
2. `t_sim = teacher_img @ teacher_txt.T`
3. KL divergence로 row-wise 정렬: `affinity_kl_rows(s_sim, t_sim, temp)`
4. `selective=True`이면 학생 margin이 낮은 행만 distill (`distill_margin_thr`)

**Ensemble Teacher 경로**:
1. Teacher별 similarity matrix 계산
2. Prior weight matrix 생성 (모드에 따라)
3. Adaptive weight 계산 (활성화 시)
4. Teacher별 per-row KL loss 계산: `affinity_kl_per_row()`
5. `_weighted_row_loss()`로 teacher 가중 합산

`affinity_columns=true`이면 column 방향 distillation도 추가합니다 (텍스트→이미지 방향).

### 7.5 Temperature Schedule

```python
_scheduled_distill_temp(epoch_idx, total_epochs, start, end, schedule)
```

| Schedule | 동작 |
|----------|------|
| `constant` | `start` 고정 |
| `linear` | `start` → `end` 선형 |
| `cosine` | `start` → `end` cosine 곡선 |

기본: `start=0.12` (softer targets) → `end=0.07` (sharper targets), cosine

### 7.6 Offline/Online 통합 인터페이스

```python
get_teacher_output(teacher, imgs, metas, offline_teacher_embs, device)
```

- `offline_teacher_embs`가 있으면 → 그대로 반환 (teacher 모델 불필요)
- 없으면 → `teacher(imgs, raw_captions)` 실행

---

## 8. 손실 함수

> 코드: `src/lpcvc_retrieval/losses.py`

### 8.1 SigLIP Loss (기본 대조 손실)

```python
siglip_loss(img_emb, txt_emb, logit_scale, logit_bias, image_ids)
```

- **출처**: [SigLIP 논문](https://arxiv.org/abs/2303.15343) — Sigmoid 기반 pairwise loss
- Softmax와 달리 **각 쌍을 독립적으로 sigmoid**로 판정
- `image_id` 기반 positive mask → `2 * mask - 1` (positive=+1, negative=-1)
- `loss = -logsigmoid(labels * logits).sum() / B`
- 작은 배치에서 Softmax보다 안정적이며 Recall 향상에 유리

### 8.2 Multi-GT Masked Contrastive Loss

```python
multi_gt_masked_contrastive_loss(img_emb, txt_emb, image_ids, logit_scale)
```

- COCO 1:5 구조에서 같은 `image_id`를 가진 모든 쌍을 정답으로 처리
- Soft target (행 정규화) + KL divergence
- False Negative 패널티 제거

### 8.3 Pairwise Ranking Loss

```python
pairwise_ranking_loss(img_emb, txt_emb, logit_scale, k=3, margin=0.1)
```

- 배치 내 top-K hard negative와 positive 간 margin을 강제
- `loss = relu(margin - (diag - topk)).mean()`

### 8.4 Hard Negative Contrastive Loss

```python
hard_negative_contrastive_loss(img_emb, txt_emb, logit_scale, num_hard_negatives=5)
```

- **출처**: BLIP, FG-CLIP
- 가장 혼동되는 negative만 선택하여 cross entropy
- I2T + T2I 양방향 대칭

### 8.5 Text-Text Contrastive Loss

```python
text_text_contrastive_loss(txt_emb, image_ids, logit_scale)
```

- **출처**: [TULIP](https://arxiv.org/abs/2406.06512)
- 같은 `image_id`의 캡션 = positive pair
- 텍스트 임베딩이 같은 의미의 다양한 표현을 가까이 매핑하도록 학습
- Self-similarity (대각선) 제외

### 8.6 기본 손실 비중

| 손실 | Config key | 기본 비중 | 출처 |
|------|-----------|----------|------|
| SigLIP loss | `w_contrastive` | 1.0 | SigLIP |
| Pairwise ranking | `w_rank` | 0.1 | — |
| Hard negative contrastive | `w_hard_negative` | 0.1 | BLIP/FG-CLIP |
| Text-text contrastive | `w_text_text` | 0.05 | TULIP |
| Affinity distillation | `w_distill_affinity` | 0.8 | — |

---

## 9. 평가 시스템

> 코드: `src/lpcvc_retrieval/metrics.py`

### 9.1 기본 Recall@K

```python
recall_at_k(sim, ks=[1, 5, 10])
```

similarity matrix에서 각 query의 정답 rank를 계산하여 Recall@K를 반환합니다.

### 9.2 Bidirectional Recall

```python
bidirectional_recall(image_emb, text_emb, ks=[1, 5, 10])
```

I2T (`image @ text.T`) + T2I (`text @ image.T`) 양방향 Recall을 계산합니다.

### 9.3 COCO-style Evaluation

```python
coco_bidirectional_recall(unique_image_emb, text_emb, unique_image_ids, text_image_ids)
```

- **이미지 중복 제거**: 같은 `image_id`의 이미지 임베딩을 하나만 사용
- **I2T**: 각 unique 이미지에 대해, 매칭되는 캡션 중 **어떤 것이든** top-K에 있으면 hit
- **T2I**: 각 캡션에 대해, 해당 이미지가 top-K에 있으면 hit
- **chunk_size**: OOM 방지용 chunk 처리 (기본 256)

### 9.4 Best Checkpoint 기준

**I2T R@10** — 매 epoch 평가 후, 최고 I2T R@10 달성 시 `best.pt` 저장

---

## 10. 학습 루프

> 코드: `src/lpcvc_retrieval/train.py`

### 10.1 `train(cfg)` 함수 흐름

```text
1. Device 결정
2. Seed 설정
3. MobileCLIP2Student 생성 → (선택) torch.compile
4. WandB 로거 초기화
5. 데이터셋/로더 생성
6. Optimizer (AdamW) + Scheduler (warmup + cosine) 생성
7. DistillConfig 파싱 → teacher 로드 (online) 또는 offline 확인
8. EMA 초기화
9. Epoch 루프:
   a. Epoch별 affinity temperature 계산
   b. 배치 루프:
      - Forward (학생)
      - SigLIP loss + (선택) ranking + hard_negative + text_text
      - Teacher output (online forward 또는 offline 읽기)
      - Affinity distill loss
      - Backward + gradient clipping + optimizer step + scheduler step
      - logit_scale clamp
      - EMA update
   c. Checkpoint 저장 (last, epoch_*)
   d. (선택) EMA 가중치로 평가 → best.pt 갱신
```

### 10.2 Scheduler

```python
get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.0)
```

- **Warmup**: 0에서 base_lr까지 선형 증가
- **Cosine decay**: base_lr에서 `min_lr_ratio * base_lr`까지 cosine 감소
- **근거**: 랜덤 초기화된 가중치에서 큰 LR은 불안정, warmup이 필수. Cosine decay는 학습 후반부 fine-grained 최적화

### 10.3 AMP

```python
torch.amp.autocast(device_type="cuda", enabled=use_amp)
torch.amp.GradScaler("cuda", enabled=use_amp)
```

CUDA에서 기본 활성. mixed precision으로 학습 속도 향상 + 메모리 절감.

### 10.4 Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
```

기본값 `grad_clip=1.0`. gradient explosion 방지.

### 10.5 logit_scale Clamp

```python
model.logit_scale.data.clamp_(logit_scale_min, logit_scale_max)
```

logit_scale이 너무 크거나 작아지는 것을 방지합니다. 기본 범위: `[-4.6, 4.6]`

---

## 11. EMA (Exponential Moving Average)

> 코드: `src/lpcvc_retrieval/ema.py`

### 11.1 원리

```
ema_weight = decay * ema_weight + (1 - decay) * current_weight
```

- **출처**: Polyak averaging (1992), CLIP/Stable Diffusion/GPT 등에서 사용
- Shadow copy를 유지하며, 매 스텝 업데이트
- 기본 `decay=0.999` → 99.9% 이전 가중치 + 0.1% 새 가중치

### 11.2 EMA 클래스 API

| 메서드 | 역할 |
|--------|------|
| `update()` | 매 optimizer step 후 shadow 업데이트 |
| `apply_shadow()` | 평가 시 shadow 가중치를 모델에 적용 |
| `restore()` | 평가 후 원래 가중치 복구 |
| `state_dict()` / `load_state_dict()` | 체크포인트 저장/로드 |

### 11.3 학습 흐름에서의 위치

```text
optimizer.step()
scheduler.step()
logit_scale.clamp_()
ema.update()          # ← 매 스텝
...
ema.apply_shadow()    # ← 평가 시작
evaluate(model)
ema.restore()         # ← 평가 종료
```

---

## 12. 로거

> 코드: `src/lpcvc_retrieval/logger.py`

### 12.1 TrainLogger

WandB와 print()를 통합하는 선택적 로거입니다.

| 메서드 | 역할 |
|--------|------|
| `log(metrics, step)` | 스텝 단위 메트릭 로깅 |
| `log_epoch(epoch, metrics)` | 에폭 단위 메트릭 로깅 |
| `finish()` | WandB run 종료 |

`use_wandb=False`(기본값)이면 WandB 호출은 전부 무시됩니다. WandB가 설치되지 않아도 에러 없이 동작합니다.

---

## 13. Export 및 배포

### 13.1 ONNX 분리 Export

> 코드: `src/lpcvc_retrieval/export.py`, `scripts/export_onnx_split.py`

#### `export_onnx_split(model, out_dir, opset, img_name, txt_name)`

내부적으로 `_ImageEnc`와 `_TextEnc` 래퍼를 생성하여 각각 독립 ONNX로 export합니다.

| 출력 파일 | 입력 | 출력 |
|----------|------|------|
| `image_encoder.onnx` | `image` float32 `(1,3,224,224)` | `embedding` float32 `(1,D)` |
| `text_encoder.onnx` | `text` int32 `(1,77)` | `embedding` float32 `(1,D)` |

- `opset`: 기본 18
- `do_constant_folding=True`
- `dynamic_axes=None` (고정 배치 크기)

#### Export 스크립트 CLI

```bash
python scripts/export_onnx_split.py --config config.yaml --ckpt best.pt --out_dir exported_onnx
```

`--prefix`와 `--add_timestamp` 옵션으로 파일명 커스터마이즈 가능.

### 13.2 Qualcomm AI Hub

> 코드: `compile_and_profile.py` (123줄)

#### `compile_model(model, device, input_specs, name)`

`qai_hub.submit_compile_job()`으로 ONNX를 QNN DLC로 컴파일합니다.
- `--target_runtime qnn_dlc --truncate_64bit_io`

#### `run_profile(compiled_model, device, name)`

`qai_hub.submit_profile_job()`으로 프로파일링합니다.
- `--max_profiler_iterations 100`

#### CLI

```bash
qai-hub configure --api_token <TOKEN>  # 최초 1회
python compile_and_profile.py --onnx_dir exported_onnx --device "XR2 Gen 2 (Proxy)"
```

---

## 14. Offline Teacher Feature

> 코드: `scripts/extract_features.py`

### 14.1 추출 흐름

1. Teacher 로드 (online)
2. train/val DataLoader를 `shuffle=False`로 순회
3. 각 배치에서 teacher forward → 임베딩 수집
4. `caption_indices`, `dataset_fingerprint` 함께 저장

### 14.2 출력 파일 형식

각 `teacher_{idx}_{split}.pt` 파일은 다음을 포함합니다:

| 키 | 타입 | 설명 |
|----|------|------|
| `img_embs` | Tensor `[N, D]` | Teacher 이미지 임베딩 |
| `txt_embs` | Tensor `[N, D]` | Teacher 텍스트 임베딩 |
| `caption_indices` | Tensor `[N]` | 사용된 캡션 인덱스 |
| `sample_count` | int | 샘플 수 |
| `dataset_fingerprint` | str | `sha1(image_id|img_rel)` 해시 |

### 14.3 `dataset_fingerprint` 생성 원리

```python
hasher = hashlib.sha1()
for each sample:
    hasher.update(f"{image_id}|{img_rel}\n".encode("utf-8"))
fingerprint = hasher.hexdigest()
```

데이터 순서/내용이 바뀌면 fingerprint가 달라져, stale feature 사용을 감지합니다.

### 14.4 운영 팁

- `data.train_augment=false`로 추출 → 재현성 확보
- Offline 모드에서는 teacher 모델을 로드하지 않음 → VRAM 0
- `--fp32` 옵션으로 fp32 저장 가능 (기본 fp16)

---

## 15. 엔트리포인트

### 15.1 `run_train.py` (88줄)

학습 엔트리포인트입니다.

- `src/`를 `sys.path`에 추가 → `PYTHONPATH=src` 없이 실행 가능
- `--config`, `--override` 인자 처리
- 디바이스 정보, 모델 variant, 에폭 등을 배너로 출력
- `train(cfg)` 호출

### 15.2 `scripts/eval.py` (98줄)

체크포인트 평가 스크립트입니다.

- `image_id`가 모두 있으면 → COCO-style 양방향 평가
- 없으면 → index 기반 bidirectional recall
- `PYTHONPATH=src` 필요

### 15.3 `scripts/export_onnx_split.py` (56줄)

ONNX 분리 export 스크립트입니다. 13.1 참조.

### 15.4 `scripts/extract_features.py` (192줄)

Offline teacher feature 추출. 14장 참조.

---

## 16. 프로젝트 구조

```text
.
├── README.md                        # 실행 가이드
├── LICENSE                          # MIT License
├── THIRD_PARTY_LICENSES.md          # 외부 모델/데이터 라이선스 정보
├── config.yaml                      # 학습 설정
├── pyproject.toml                   # 패키지 메타 (setuptools, src layout)
├── requirements.txt                 # pip 의존성
├── run_train.py                     # 학습 엔트리포인트
├── compile_and_profile.py           # QAI Hub 컴파일/프로파일
│
├── docs/
│   ├── README.md                    # 문서 인덱스
│   ├── PROJECT_GUIDE.md             # 이 파일
│   └── archive/                     # 완료된 기록
│
├── scripts/
│   ├── eval.py                      # 체크포인트 평가
│   ├── export_onnx_split.py         # ONNX 분리 export
│   ├── extract_features.py          # Offline teacher feature 추출
│   └── preprocess/
│       ├── parse_lpcvc_sources.py   # 4개 데이터셋 → JSONL 통합
│       └── materialize_upload_subset.py  # 업로드용 서브셋 생성
│
└── src/lpcvc_retrieval/
    ├── __init__.py
    ├── config.py                    # CfgNode, load_config, resolve_device
    ├── data.py                      # 데이터셋, 증강, collate_fn
    ├── distill.py                   # Teacher, routing, affinity distillation
    ├── ema.py                       # EMA 구현
    ├── export.py                    # ONNX export 유틸
    ├── logger.py                    # WandB 로거
    ├── losses.py                    # SigLIP, ranking, hard-neg, text-text
    ├── metrics.py                   # Recall@K, COCO-style 평가
    ├── mobileclip2.py               # MobileCLIP2 학생 래퍼
    ├── model.py                     # 모델 팩토리, OnnxWrapper
    └── train.py                     # 학습 루프, evaluate, scheduler
```

---

## 17. 라이선스 및 출처

### 17.1 저장소 라이선스

이 저장소의 코드는 **MIT License**입니다 (`LICENSE` 파일).

### 17.2 모델 및 프레임워크

| 구성요소 | 출처 | 라이선스 |
|----------|------|---------|
| MobileCLIP2 코드 | [apple/ml-mobileclip](https://github.com/apple/ml-mobileclip) | MIT |
| MobileCLIP2 가중치 | [apple/MobileCLIP2-S0](https://huggingface.co/apple/MobileCLIP2-S0) | Apple model card terms |
| SigLIP2 teacher | [timm/ViT-gopt-16-SigLIP2-256](https://huggingface.co/timm/ViT-gopt-16-SigLIP2-256) | Apache-2.0 |
| PE-Core teacher | [facebook/PE-Core-G14-448](https://huggingface.co/facebook/PE-Core-G14-448) | Apache-2.0 |
| OpenCLIP | [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) | MIT |
| CLIP Tokenizer | [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) | MIT |

### 17.3 데이터셋

| 데이터셋 | 출처 | 라이선스 |
|----------|------|---------|
| COCO 2017 | [cocodataset.org](https://cocodataset.org/) | COCO terms |
| Flickr30k | [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr30k) | 제공자 약관 |
| Open Images V7 | [Google](https://storage.googleapis.com/openimages/web/index.html) | Open Images terms |
| WIT | [google-research-datasets/wit](https://github.com/google-research-datasets/wit) | WIT terms |

### 17.4 학술 참고

| 기법 | 논문/출처 |
|------|----------|
| SigLIP Loss | [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) |
| Hard Negative Mining | [BLIP](https://arxiv.org/abs/2201.12086), [FG-CLIP](https://arxiv.org/abs/2405.11510) |
| Text-Text Contrastive | [TULIP](https://arxiv.org/abs/2406.06512) |
| EMA | Polyak averaging (1992) |
| Knowledge Distillation | [TinyCLIP](https://arxiv.org/abs/2309.12314) |
| Cosine LR Schedule | [SGDR](https://arxiv.org/abs/1608.03983) |

> 재배포 또는 상업 사용 전 각 모델 카드와 데이터셋 약관을 반드시 재확인하세요.

---

## 18. 코드 읽기 추천 순서

처음 읽는다면 아래 순서가 가장 빠릅니다:

1. `run_train.py` — 엔트리포인트, 전체 파이프라인 개요
2. `src/lpcvc_retrieval/train.py` — 학습 루프 전체
3. `src/lpcvc_retrieval/data.py` — 데이터 파이프라인
4. `src/lpcvc_retrieval/mobileclip2.py` — 학생 모델
5. `src/lpcvc_retrieval/distill.py` — teacher + distillation 핵심
6. `src/lpcvc_retrieval/losses.py` — 손실 함수 모음
7. `src/lpcvc_retrieval/metrics.py` — 평가 지표
8. `scripts/eval.py` — 평가 실행
9. `scripts/export_onnx_split.py` — ONNX export
10. `compile_and_profile.py` — QAI Hub 배포

---

## 19. 문서 정리 원칙

| 문서 | 역할 |
|------|------|
| `README.md` | 실행 중심 |
| `docs/PROJECT_GUIDE.md` | 구조, 구현 중심 (이 파일) |
| `docs/archive/` | 완료된 handover 및 정리 기록 |

새 실험 추가 시 같이 갱신해야 하는 파일: `config.yaml`, `README.md`, `docs/PROJECT_GUIDE.md`

특히 다음 중 하나가 바뀌면 문서를 같이 갱신:
- Teacher 모델 이름
- Distillation 기본값
- 데이터 형식
- Export 입력 형상
- QAI Hub 사용 방식
