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
