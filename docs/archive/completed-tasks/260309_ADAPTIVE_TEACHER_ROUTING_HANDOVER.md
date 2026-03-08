# Adaptive Teacher Routing 코드 수정 인수인계서

기준일: 2026-03-09  
대상 변경: `config.yaml`, `src/lpcvc_retrieval/data.py`, `src/lpcvc_retrieval/distill.py`, `src/lpcvc_retrieval/train.py`

## 1. 이번 수정의 목적

기존 teacher distillation 경로는 정적 teacher weight 중심으로 해석되기 쉬웠습니다.  
이번 수정의 목적은 아래 두 가지입니다.

1. teacher 2개를 항상 같은 방식으로 섞지 않고, 현재 배치와 샘플 품질을 보고 더 적절한 teacher 비중을 사용하도록 만들기
2. 향후 데이터셋 source별 routing 실험이 가능하도록, 데이터 메타와 distillation 경로를 확장하기

현재 기본 방향은 `source prior 없이 순수 adaptive routing`입니다.

## 2. 변경 결과 한 줄 요약

- 기본 distillation 동작이 `adaptive_teacher_weight: true` + `teacher_weight_mode: adaptive`로 바뀌었습니다.
- `source-aware routing` 기능은 코드에 추가했지만, 현재 기본값으로는 사용하지 않습니다.
- 데이터셋의 `source` 값이 학습 메타로 전달되도록 수정했습니다.

## 3. 파일별 변경 내용

### 3.1 `config.yaml`

추가 및 변경된 설정:

```yaml
distill:
  adaptive_teacher_weight: true
  adaptive_teacher_tau: 0.07
  adaptive_teacher_w_min: 0.20
  static_teacher_weights: [0.5, 0.5]
  teacher_weight_mode: adaptive
  source_teacher_weights: {}
```

의미:
- `adaptive_teacher_weight: true`
  - teacher mixing을 동적으로 계산하도록 활성화
- `adaptive_teacher_tau`
  - adaptive softmax temperature
- `adaptive_teacher_w_min`
  - 한 teacher로 weight가 완전히 몰리는 현상 완화
- `static_teacher_weights`
  - `teacher_weight_mode=static`일 때만 쓰는 고정 mixing 비율
- `teacher_weight_mode: adaptive`
  - 현재 기본 동작은 source prior 없는 adaptive routing
- `source_teacher_weights: {}`
  - source prior는 비워 둠

정리:
- teacher 정의(`teachers`)와 mixing 정책(`static_teacher_weights`)을 분리했습니다.
- 현재 기본 모드(`adaptive`)에서는 `static_teacher_weights`가 최종 mixing 비율에 영향을 주지 않습니다.

### 3.2 `src/lpcvc_retrieval/data.py`

추가된 핵심:
- `_clean_source()` 함수 추가
- JSONL dataset에서 `source` 필드를 읽어 정리
- COCO dataset도 `source="coco"`를 메타에 포함
- 최종 `meta`에 `source` 추가

변경 효과:
- 학습 루프가 각 샘플의 데이터 출처를 알 수 있게 됨
- 이후 `adaptive_source`를 켜면 source별 prior routing을 적용할 수 있음

현재 메타 필드:
- `image_id`
- `ann_id`
- `caption`
- `caption_idx`
- `img_rel`
- `source`

### 3.3 `src/lpcvc_retrieval/distill.py`

가장 큰 수정이 들어간 파일입니다.

추가된 설정 필드:
- `teacher_weight_mode`
- `source_teacher_weights`

핵심 변경:

1. teacher 품질 점수를 `배치 전체 1개 값`이 아니라 `샘플별 row margin`으로 계산
2. teacher weight를 `teacher x row` 형태의 matrix로 다룰 수 있게 확장
3. `adaptive_source` 모드를 위한 source prior 결합 함수 추가
4. per-row KL loss를 계산해 teacher별 가중합이 가능하도록 변경
5. `adaptive` 모드에서는 uniform prior에서 시작하도록 정리
6. `teachers[].weight`를 제거하고 `static_teacher_weights`로 책임 분리

새로 들어간 주요 함수:
- `_teacher_quality_margin_per_row(...)`
- `_resolve_source_prior_weights(...)`
- `_combine_teacher_weights(...)`
- `_normalize_weight_matrix(...)`
- `affinity_kl_per_row(...)`
- `_build_prior_weight_matrix(...)`
- `_weighted_row_loss(...)`

현재 adaptive score 정의:
- 각 샘플 row 기준
- `positive similarity 평균 - hardest negative similarity`

즉, retrieval 관점에서 같은 이미지에 해당하는 positive를 잘 올리고, 가장 어려운 negative를 잘 밀어낸 teacher에 더 큰 weight를 주는 방식입니다.

### 3.4 `src/lpcvc_retrieval/train.py`

변경 내용:
- `teacher_weight_mode` 읽기 및 유효성 검사 추가
- `source_teacher_weights` 읽기 추가
- `adaptive_teacher_weight=True`인데 `teacher_weight_mode=static`이면 자동 승격
  - source prior가 없으면 `adaptive`
  - source prior가 있으면 `adaptive_source`
- 학습 배치의 `meta["source"]`를 읽어 `sample_sources` 생성
- distill loss 호출 시 아래 인자 추가 전달
  - `teacher_weight_mode`
  - `source_teacher_weights`
  - `sample_sources`

부가 변경:
- 현재 실행 로그에 `teacher_weight_mode`가 출력됨
- `adaptive_source`일 때 source key 목록도 출력됨

## 4. 현재 기본 동작

현재 기본 config 기준 실제 동작:

1. teacher 두 개를 모두 로드하거나, offline feature를 읽음
2. 학생 similarity matrix 계산
3. teacher별 similarity matrix 계산
4. 각 샘플 row에서 어떤 teacher가 더 좋은 margin을 내는지 계산
5. 그 결과로 teacher weight를 동적으로 생성
6. affinity distillation loss를 row-wise 가중합

현재 기본값은 아래와 같습니다.

```yaml
distill:
  adaptive_teacher_weight: true
  teacher_weight_mode: adaptive
  source_teacher_weights: {}
```

의미:
- 현재는 `source-aware prior` 없이
- teacher 품질만 보고 동적으로 mixing
- `static_teacher_weights`는 현재 기본 모드에서 영향 없음

## 5. 왜 이렇게 바꿨는가

이번 결정의 이유는 다음과 같습니다.

1. 실제 평가 분포가 명확히 공개되지 않은 상태에서 source prior를 강하게 넣는 것은 가정에 불과함
2. teacher 2개가 각자 다른 강점을 가지므로, 고정 mixing보다 adaptive routing이 더 합리적임
3. source-aware 기능은 실험 옵션으로 남겨두되, 기본값은 편향이 적은 방향으로 두는 것이 안전함

즉, 현재 기본 설정은:
- teacher를 둘 다 활용하되
- 어느 teacher가 항상 더 낫다고 미리 가정하지 않는 baseline

## 6. 아직 남아 있는 선택지

코드에는 아래 세 모드가 모두 남아 있습니다.

- `static`
- `adaptive`
- `adaptive_source`

현재 권장 baseline:
- `adaptive`

실험 옵션:
- source별 prior를 명시하고 싶으면 `adaptive_source`

예시:

```yaml
distill:
  adaptive_teacher_weight: true
  teacher_weight_mode: adaptive_source
  source_teacher_weights:
    coco:
      ViT-gopt-16-SigLIP2-256: 0.6
      PE-Core-bigG-14-448: 0.4
```

다만 현재 인수인계 기준에서는 source prior를 넣지 않는 상태가 기본값입니다.

## 7. 검증 상태

확인된 사항:
- `py_compile` 기준 문법 오류 없음
- 실제 JSONL source 키 전달 경로 구현됨
- `config.yaml` 기본값이 adaptive 모드로 반영됨

아직 실험으로 확인해야 하는 것:
- `static` 대비 `adaptive`의 실제 `I2T/T2I R@10` 향상 여부
- online teacher 대비 offline feature 모드의 속도/성능 차이
- `adaptive_source`가 순수 adaptive보다 일관되게 좋은지 여부

## 8. 운영 시 주의사항

1. `teacher_weight_mode`를 `adaptive_source`로 바꾸더라도 `source_teacher_weights`가 비어 있으면 실질 prior는 없음
2. `source` 필드가 없는 JSONL은 `unknown`으로 처리됨
3. `static_teacher_weights`는 `static` 모드에서만 의미가 있음
4. `adaptive_source`에서 source key가 빠진 샘플은 uniform prior로 처리됨
5. 현재 활성 문서는 이 변경을 반영하도록 업데이트됨

## 9. 다음 작업 권장 순서

1. 현재 baseline 그대로 `adaptive` 1회 학습
2. 같은 조건으로 `static` 1회 학습
3. `I2T/T2I R@10`, 학습 시간, GPU 메모리 비교
4. 그 다음에만 `adaptive_source` 실험 진행

## 10. 결론

이번 수정의 핵심은 `teacher 두 개를 정적으로 다루는 코드`에서 `샘플별 adaptive teacher routing이 가능한 코드`로 확장한 것입니다.

현재 기본값은:
- source prior 없음
- 순수 adaptive routing
- 듀얼 teacher 유지

즉, 현재 저장소의 teacher distillation baseline은 `adaptive teacher weighting`입니다.
