# MobileNetV4 통합 변경 로그

**날짜**: 2026-01-26  
**버전**: v3.0 (MobileNetV4 Migration)

---

## 개요

FastViT-S12에서 **MobileNetV4-Medium**으로 Vision Backbone을 변경했습니다.  
이 변경은 LPCVC 2026 대회의 Qualcomm Hexagon DSP/NPU 최적화를 위해 수행되었습니다.

---

## 변경 사항

### 1. config.yaml

#### Vision Backbone 변경
```diff
- vision_backbone: fastvit_s12.apple_in1k
+ vision_backbone: mobilenetv4_conv_medium.e500_r224_in1k
```

#### Normalization 변경 (ImageNet 표준)
```diff
  clip_mean:
-  - 0.48145466
-  - 0.4578275
-  - 0.40821073
+  - 0.485
+  - 0.456
+  - 0.406
  clip_std:
-  - 0.26862954
-  - 0.26130258
-  - 0.27577711
+  - 0.229
+  - 0.224
+  - 0.225
```

### 2. src/lpcvc_retrieval/model.py

#### VisionTower 호환성 개선
- `feat.flatten(1)` 추가: MobileNetV4 출력이 `[B,C,1,1]` 형태일 수 있으므로 `[B,C]`로 변환
- `feat_dim` 추론을 항상 수행하도록 변경

```python
# Before
feat = self.backbone(x)  # [B,C]
emb = self.proj(feat)

# After
feat = self.backbone(x)  # [B,C] or [B,C,1,1]
feat = feat.flatten(1)   # Force [B,C] for MobileNetV4 compatibility
emb = self.proj(feat)
```

### 3. run_train.py (신규)

CMD에서 쉽게 학습을 실행할 수 있는 스크립트 추가:
```cmd
python run_train.py --config config.yaml
```

- 학습 시작/종료 시간 표시
- 설정 정보 출력 (backbone, epochs, batch_size, lr)

### 4. scripts/train.py (삭제)

`run_train.py`와 기능 중복으로 제거함.

### 5. src/lpcvc_retrieval/distill.py

Teacher 모델 로드 시 `use_safetensors=True` 적용:
```python
self.model = CLIPModel.from_pretrained(hf_id, use_safetensors=True)
```
- pickle 직렬화 취약점 방지 (보안 강화)

---

## 변경하지 않은 항목

- **SigLIP Loss**: 기존 Sigmoid 기반 Multi-GT 손실 함수 유지
- **커리큘럼 학습**: 추가하지 않음 (SigLIP과 함께 사용 시 효과 제한적)
- **scripts/**: `eval.py`, `export_onnx_split.py` 등은 config에서 동적으로 모델 설정을 읽으므로 수정 불필요

---

## MobileNetV4 선택 이유

| 항목 | FastViT-S12 | MobileNetV4-Medium |
|------|-------------|-------------------|
| **출시 시기** | 2023년 3월 (Apple) | 2024년 4월 (Google) |
| 파라미터 수 | ~8M | ~9M |
| ImageNet Top-1 | ~79% | ~80% |
| **추론 속도 (Hexagon)** | 기준 | **~2배 빠름** |
| ONNX 호환성 | ⚠️ 일부 연산 주의 | ✅ 완벽 지원 |
| NPU 가속 최적화 | - | **UIB 블록 설계** |

---

## 호환성 확인

### 대회 규정 준수
- ✅ 이미지 입력: `float32 [0,1]`, `1×3×224×224`
- ✅ 텍스트 입력: `int32`, `1×77`
- ✅ 토크나이저: `openai/clip-vit-base-patch32`
- ✅ 정규화: 모델 내부 처리 (`normalize_input: true`)

---

## 실행 방법

### 학습
```cmd
python run_train.py --config config.yaml
```

### 평가
```cmd
python scripts/eval.py --config config.yaml --ckpt runs/lpcvc_clip_lite/best.pt
```

### ONNX Export
```cmd
python scripts/export_onnx_split.py --config config.yaml --ckpt runs/lpcvc_clip_lite/best.pt --out_dir exported_onnx
```

---

## 예상 성능

| 메트릭 | 예상 범위 |
|--------|----------|
| I2T R@10 | 60-65% |
| T2I R@10 | 55-60% |
| 추론 속도 | FastViT 대비 ~2배 향상 |

---

## 2026-01-30: 코드 리팩토링 및 환경 개선 (Refactoring & Optimization)

@python-pro, @ml-engineer, @debugger, @code-reviewer 스킬을 활용하여 프로젝트 전반의 코드 품질과 실행 환경을 개선했습니다.

### 1. 주요 변경 사항
- **Editable Install 도입**: `pip install -e .`를 사용하여 패키지를 설치하도록 변경. `sys.path` 조작 코드 제거.
- **WandB 모듈화**: `src/lpcvc_retrieval/logger.py`로 로깅 로직 분리.
- **DataLoader 최적화**: `persistent_workers=True` 추가로 Epoch 시작 속도 개선.
- **torch.compile 지원**: PyTorch 2.x 컴파일 기능 추가 (학습 가속).
- **코드 품질**: 불필요한 `if True:` 제거 및 PEP 8 준수 Import 정리.

### 2. WandB (Weights & Biases) 사용 가이드

실험 추적을 위해 WandB를 사용할 수 있습니다 (선택 사항).

#### 1) 사전 준비
```bash
pip install wandb
wandb login
# API Key 입력 (https://wandb.ai/authorize 에서 확인)
```

#### 2) 활성화 방법 (`config.yaml`)
WandB를 사용하려면 `config.yaml`의 `train` 섹션을 수정하세요.

```yaml
train:
  use_wandb: true                  # WandB 사용 여부 (기본값: false)
  wandb_project: "lpcvc-clip-lite" # WandB 프로젝트 이름
  wandb_run_name: "mobilenetv4-v1" # (선택) 실험 이름 (비워두면 자동 생성)
```

#### 3) 로깅되는 정보
- **Step 단위**: Loss, Learning Rate
- **Epoch 단위**: Validation R@10 (I2T, T2I)
- **제거 방법**: `config.yaml`에서 `use_wandb: false`로 설정하거나 `logger.py` 삭제 후 `train.py` 수정
