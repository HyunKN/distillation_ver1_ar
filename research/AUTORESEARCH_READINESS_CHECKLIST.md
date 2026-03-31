# Autoresearch Readiness Checklist

작성일: 2026-04-01

이 문서는 `autoresearch`를 실제로 돌리기 전에 무엇을 먼저 확정해야 하는지 정리한 체크리스트입니다.

핵심 원칙:

- 먼저 `무엇을 최적화할지`를 정한다.
- 그 다음 `무엇을 바꿔도 되는지`를 정한다.
- 그 다음 `어떻게 기록하고 채택할지`를 정한다.
- 마지막으로 `서버 하네스와 실행 하네스`를 만든다.

---

## Phase 0. Metric Readiness

### 0-1. 대회 metric 정의 재확인

상태: TODO

확인할 것:

- Track 1 품질 metric이 정확히 어떤 정의의 `Recall@Top10`인지
- 현재 내부 `I2T R@10`과 완전히 같은지
- `정답 텍스트 하나 hit` 방식인지
- `top 10 안의 GT text 개수 / 전체 GT text 개수` 방식인지

현재 판단:

- 현재 코드의 `I2T R@10`은 `정답 중 하나라도 top-10에 있으면 hit`에 가깝다.
- 대회 설명식과는 정확히 같지 않을 가능성이 있다.

### 0-2. 평가 코드 수정 여부 결정

상태: TODO

선택지:

- 기존 metric 유지 + 대회식 metric 추가
- 대회식 metric으로 완전 교체

권장:

- 기존 metric은 내부 비교용으로 유지
- 대회식 metric을 추가
- best checkpoint는 대회식 metric 기준으로 선택

---

## Phase 1. Research Contract

### 1-1. 최종 목표 metric 확정

상태: TODO

권장안:

- Primary metric: `competition_I2T_R@10`
- Secondary metrics:
  - `mean_R@10`
  - `T2I_R@10`
  - `R@1`
  - `R@5`
- Later hard constraint:
  - compiled latency `< 35ms`

### 1-2. 수정 허용 범위 확정

상태: TODO

권장안:

- 자동 허용:
  - `config.yaml`
  - `configs/subset_single_teacher_search.yaml`
  - `--override`로 바꿀 수 있는 train/loss/distill/data 값
- 승인 후 허용:
  - `src/lpcvc_retrieval/train.py`
  - `src/lpcvc_retrieval/losses.py`
  - `src/lpcvc_retrieval/distill.py`
- 별도 설계 승인 후:
  - `src/lpcvc_retrieval/dual_tower.py`
  - `src/lpcvc_retrieval/model.py`

### 1-3. 수정 금지 범위 확정

상태: TODO

권장안:

- `src/lpcvc_retrieval/metrics.py`는 metric 수정 완료 전까지 보호
- `src/lpcvc_retrieval/data.py`는 데이터 정의 보호
- `scripts/eval.py`는 평가 기준 확정 전까지 함부로 변경 금지
- 배포 관련 코드는 현재 탐색 루프에서 제외

### 1-4. 로그 형식 확정

상태: TODO

권장안:

- W&B를 기본 실험 로그로 사용
- `research/EXPERIMENTS.tsv`는 사람이 읽는 공식 기록으로 유지
- run 당 기록:
  - run name
  - config / overrides
  - primary metric
  - secondary metrics
  - train loss
  - distill loss
  - crash / OOM / NaN 여부
  - keep / discard / retest

### 1-5. 채택 기준 확정

상태: TODO

권장안:

- `KEEP`
  - primary metric 개선
  - crash 없음
  - 결과 해석 가능
- `RETEST`
  - 개선은 있지만 차이가 작음
  - variance 확인 필요
- `DISCARD`
  - 개선 없음
  - 불안정
- `CRASH`
  - OOM / NaN / eval 실패

---

## Phase 2. Server Harness Design

### 2-1. 서버 경로 구조 확정

상태: TODO

예시:

- repo: `/home/<user>/workspace/distillation_ver1_ar`
- images: `/data/lpcvc/images`
- jsonl: `/data/lpcvc/prepared_jsonl`
- subset: `/data/lpcvc/prepared_jsonl_subset`
- runs: `/data/lpcvc/runs`
- hf cache: `/data/lpcvc/cache/huggingface`

### 2-2. 실행 방식 확정

상태: TODO

권장안:

- first loop: `subset search`
- second loop: `full verify`
- last loop: `compile/profile`

### 2-3. W&B 운영 규칙 확정

상태: TODO

정할 것:

- project name
- run naming rule
- tags rule
- group rule

---

## 현재 우선순위

지금 당장 먼저 할 것:

1. 대회 metric과 내부 metric 차이 확정
2. 평가 코드 수정 방향 결정
3. 최종 목표 metric 확정
4. 수정 허용 범위 / 금지 범위 확정
5. 로그 형식 / 채택 기준 확정
6. 그 다음 서버 하네스 설계

아직 나중으로 미뤄도 되는 것:

- 최종 배포 자동화
- latency 최적화 루프
- 구조 변경 autoresearch
- Jupyter 기반 분석
