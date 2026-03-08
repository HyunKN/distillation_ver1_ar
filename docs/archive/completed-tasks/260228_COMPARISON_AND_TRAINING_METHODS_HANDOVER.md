# 비교 컨텍스트 및 현재 학습 방식 handover

기준일: 2026-03-09  
기준: 현재 저장소 코드

이 문서는 완료된 handover 기록이므로 archive에 보관합니다.  
현재 운영 기준 문서는 `README.md`, 상세 기술 문서는 `docs/PROJECT_GUIDE.md`를 사용합니다.

## 1. 이 문서의 목적

이 문서는 아래 질문에 답하기 위한 상태 기록입니다.

- 현재 코드가 실제로 무엇을 하는가
- 최근에 무엇이 바뀌었는가
- 어떤 가정이 더 이상 맞지 않는가
- 다음 실험은 무엇이 합리적인가

## 2. 현재 기준 사실 관계

현재 코드 기준:
- 패키지 루트: `src/lpcvc_retrieval/`
- 학습 엔트리포인트: `run_train.py`
- 기본 데이터 모드: `jsonl`
- 기본 학생 모델: `MobileCLIP2-S0`
- 기본 teacher 구성: dual teacher
- 기본 teacher routing: `adaptive`
- source-aware routing은 구현되어 있지만 기본값은 아님

현재 활성 문서:
- `README.md`
- `docs/PROJECT_GUIDE.md`

## 3. 최근 바뀐 내용

### 문서와 구조

- `PROJECT_GUIDE.md`가 루트에서 `docs/PROJECT_GUIDE.md`로 이동
- handover 문서가 `docs/archive/completed-tasks/`로 이동
- README는 실행 중심 문서로 유지

### 데이터 파이프라인

- JSONL의 `source`가 런타임 메타로 전달됨
- JSONL, COCO 모두 `meta["source"]`를 반환
- offline feature 사용 시 `dataset_fingerprint`, `caption_indices`를 검증

### distillation 파이프라인

- `static`, `adaptive`, `adaptive_source` 세 가지 teacher routing 지원
- 기본값은 더 이상 정적 mixing이 아님
- adaptive routing은 배치 전체 1개 값이 아니라 샘플 단위 품질 점수를 사용

## 4. 현재 학습 스택

`src/lpcvc_retrieval/train.py` 기준:
- optimizer: `AdamW`
- scheduler: warmup + cosine
- AMP: CUDA에서 활성
- gradient clipping: 활성
- EMA: 기본 활성
- `torch.compile`: 기본 활성

현재 손실 스택:
- `siglip_loss`
- `pairwise_ranking_loss`
- `hard_negative_contrastive_loss`
- `text_text_contrastive_loss`
- `compute_affinity_distill_loss`

best 모델 선정 기준:
- `I2T R@10`

## 5. 현재 distillation 설계

### teacher

기본 teacher:
- `ViT-gopt-16-SigLIP2-256`
- `PE-Core-bigG-14-448`

### routing 모드

- `static`: config에 적힌 teacher weight 그대로 사용
- `adaptive`: 현재 배치, 샘플에서 teacher 품질을 보고 동적 mixing
- `adaptive_source`: source prior와 adaptive score를 함께 사용

### 현재 기본 설정

```yaml
distill:
  adaptive_teacher_weight: true
  teacher_weight_mode: adaptive
  source_teacher_weights: {}
```

의미:
- source prior를 넣지 않음
- 특정 데이터셋에 teacher 우열을 미리 가정하지 않음
- 현재 배치에서 어떤 teacher가 더 좋은지 보고 비중을 결정

### adaptive score의 현재 정의

teacher 품질은 각 샘플에 대해 아래 margin으로 계산합니다.

- positive similarity 평균
- hardest negative similarity

즉, `positive mean - hardest negative`가 큰 teacher를 더 신뢰하는 구조입니다.

실무 해석:
- 두 teacher를 모두 사용
- hard switch가 아니라 soft routing
- 현재 기본값은 source prior보다 덜 편향된 baseline

## 6. 왜 source prior를 기본값으로 두지 않았는가

실제 JSONL의 source 키는 확인되어 있습니다.

- `coco`
- `flickr30k`
- `open_images`
- `wit`

그럼에도 기본값에서 source prior를 쓰지 않는 이유:
- source prior는 가설
- 잘못된 prior는 teacher routing을 왜곡할 수 있음
- 평가 분포가 공개되지 않은 상황에서는 순수 adaptive가 더 안전한 baseline

source-aware routing은 코드에 남겨두고, 명시적 실험 항목으로 취급합니다.

## 7. Offline teacher feature

`scripts/extract_features.py`로 teacher feature를 사전 추출할 수 있습니다.

저장 항목:
- teacher별 이미지 임베딩
- teacher별 텍스트 임베딩
- `caption_indices`
- `sample_count`
- `dataset_fingerprint`

학습 시 검증:
- feature 길이와 데이터셋 길이 일치
- 이미지, 텍스트 임베딩 shape 일치
- teacher 간 `caption_indices` 일치
- teacher 간 `dataset_fingerprint` 일치
- 현재 데이터셋과 feature 파일의 `dataset_fingerprint` 일치

의미:
- JSONL이 바뀌었는데 오래된 feature를 쓰는 실수를 막음
- train 시 사용한 caption과 feature 추출 시 caption이 어긋나는 문제를 줄임

## 8. 데이터 계약

현재 JSONL 파서가 기대하는 한 줄 형식:

```json
{"image":"coco/train2017/000000000009.jpg","captions":["a cat on sofa"],"source":"coco"}
```

현재 런타임 메타 필드:
- `image_id`
- `ann_id`
- `caption`
- `caption_idx`
- `img_rel`
- `source`

이 메타는 아래 기능에 사용됩니다.
- multi-positive 학습
- COCO-style 평가
- offline feature 재현성 검증
- source-aware teacher routing

## 9. 더 이상 맞지 않는 오래된 가정

아래 가정은 현재 코드 기준으로 outdated 입니다.

- teacher mixing이 항상 `0.5 / 0.5`라는 가정
- source-aware routing이 기본값이라는 가정
- 문서가 모두 루트에 있다는 가정
- offline feature 파일을 무검증으로 신뢰한다는 가정

## 10. 다음 실험 우선순위

현재 기본 상태에서 우선 해볼 만한 실험:

1. `static` vs `adaptive`
- 동일 데이터
- 동일 seed
- `I2T/T2I R@10` 비교

2. online teacher vs offline feature
- `distill.offline_feature_dir`만 다르게 유지
- 속도와 metric parity 비교

3. `adaptive` vs `adaptive_source`
- source prior가 반복적으로 도움된다는 근거가 생긴 뒤에 수행
- source prior는 처음엔 약하게 시작

4. `affinity_columns=false` vs `true`
- 현재 기본값은 `false`
- 추가 연산 비용 대비 이득이 있는지 확인

## 11. 현재까지 검증된 것

이번 업데이트 사이클에서 확인한 사항:
- 주요 학습, 평가, export 파일 `py_compile` 통과
- 문서 경로가 정리된 저장소 구조와 일치
- 실제 prepared JSONL에서 source 라벨 확인

## 12. 요약

현재 저장소 방향:
- 실행 경로는 안정적으로 유지
- 문서는 코드 기준으로 맞춤
- teacher 기본 baseline은 `adaptive`
- source prior는 옵션 실험으로 취급

실제 운영은 `README.md`, 상세 구조 확인은 `docs/PROJECT_GUIDE.md`를 기준으로 진행하면 됩니다.