# Autoresearch-Lite 적용 계획

작성일: 2026-03-24

이 문서는 현재 레포에 `autoresearch-lite`를 적용할 때 무엇을 바꾸고, 무엇을 바꾸지 않을지 정리한 실행 계획서입니다.

## 1. 지금 추가하는 것

### 새 문서

- `program.md`
- `research/AUTORESEARCH_LITE_PLAN.md`

역할:

- 자동 실험의 범위와 금지선을 명확히 함
- 나중에 다른 사람이 와도 같은 규칙으로 이어갈 수 있게 함

### 새 스크립트

- `scripts/preprocess/make_jsonl_subset.py`

역할:

- 기존 `prepared_jsonl`에서 고정 subset을 생성
- 같은 subset을 반복 사용해 config 비교의 공정성을 확보

### 새 config

- `configs/subset_no_teacher_smoke.yaml`
- `configs/subset_single_teacher_smoke.yaml`
- `configs/subset_single_teacher_search.yaml`

역할:

- full data 대신 작은 고정 subset에서 빠른 smoke와 config 탐색을 수행

### 기록 확장

- `research/EXPERIMENTS.tsv`에 subset 실험 행 추가
- `research/DECISIONS.md`에 운영 결정 추가

## 2. 지금 건드리지 않는 것

### 자동 수정 금지 파일

- `src/lpcvc_retrieval/data.py`
- `src/lpcvc_retrieval/metrics.py`
- `src/lpcvc_retrieval/config.py`
- `scripts/eval.py`
- `src/lpcvc_retrieval/export.py`
- `compile_and_profile.py`

이유:

- 데이터 정의와 평가 기준이 흔들리면 실험 비교가 무의미해짐
- 지금은 배포가 아니라 baseline과 config 탐색이 목적임

### 승인 후 수정 파일

- `src/lpcvc_retrieval/train.py`
- `src/lpcvc_retrieval/losses.py`
- `src/lpcvc_retrieval/distill.py`

이유:

- 학습 의미 자체가 바뀔 수 있음
- config 탐색보다 리스크가 큼

### 구조 변경 보류 파일

- `src/lpcvc_retrieval/dual_tower.py`
- `src/lpcvc_retrieval/model.py`

이유:

- baseline 해석이 어려워짐
- 나중 배포 경로에도 영향 가능

## 3. 운영 방식

### 지금 채택

- `config-only autoresearch`
- `고정 subset + 적은 epoch 완주`
- `keep / discard / crash` 기록
- `full data 재검증 후 채택`

### 지금 보류

- full autonomous code rewriting
- 시간 예산 기반 중간 종료 학습
- 배포 자동 검증 루프

## 4. 왜 이 방식이 맞는가

- 현재 학습 코드는 epoch 끝 평가 구조라 time-budget 방식보다 subset 완주가 안전함
- RTX 2060 환경에서는 full data 반복 실험이 너무 오래 걸림
- baseline 이후에 config 값만 자동 탐색하는 것은 리스크가 낮고 성과 대비 효율이 높음
- 연구용 스크립트가 본체에 강하게 박히지 않아 최종 제출 전에 정리하기 쉬움

## 5. 최종 제출 전 제거 가능한 것

- `program.md`
- `research/AUTORESEARCH_LITE_PLAN.md`
- `research/EXPERIMENTS.tsv`
- `scripts/preprocess/make_jsonl_subset.py`
- `configs/subset_*.yaml`

즉, 지금 추가하는 autoresearch-lite 관련 파일은 대부분 `연구 보조용`이며 제출 본체와 분리됩니다.

