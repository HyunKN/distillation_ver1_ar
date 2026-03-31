# LPCVC Autoresearch-Lite Program

이 문서는 현재 레포에서 `autoresearch`의 운영 원칙만 가져와 안전하게 적용하기 위한 지시서입니다.

## 목표

- 현재 프로젝트의 주력 방향은 `SigLIP single teacher baseline`입니다.
- 지금 단계의 목표는 `설명 가능하고 재현 가능한 baseline`과 `저위험 config 탐색 체계`를 만드는 것입니다.
- 지금은 `Qualcomm AI Hub 배포`를 자동 실험 루프에 포함하지 않습니다.
- 배포 확인은 `최종 후보 모델`이 정해진 뒤에만 진행합니다.

## 현재 기본 원칙

- baseline 없이 고급 튜닝부터 시작하지 않습니다.
- 한 번에 한 가지 큰 변화만 봅니다.
- 실험은 반드시 기록합니다.
- data/metric 정의를 함부로 바꾸지 않습니다.
- 구조 변경은 자동화하지 않습니다.

## 실험 단계

### Phase A. Manual Baseline

- `configs/no_teacher_smoke.yaml`
- `configs/single_teacher_smoke.yaml`
- `config.yaml` 또는 `configs/single_teacher_baseline.yaml`

목적:

- 파이프라인 정상 동작 확인
- single teacher baseline 확보

### Phase B. Config-Only Autoresearch-Lite

- 고정 subset 데이터셋 사용
- 짧은 epoch 완주 방식 사용
- `config`와 `--override`만 자동 탐색

기본 search config:

- `configs/subset_single_teacher_search.yaml`

기본 목표:

- `val_mean_r10` 개선
- NaN/OOM/crash 없음
- baseline 대비 설명 가능한 개선만 채택

### Phase C. Human-Approved Code Changes

아래는 사람 승인 후에만 진행합니다.

- `src/lpcvc_retrieval/train.py`
- `src/lpcvc_retrieval/losses.py`
- `src/lpcvc_retrieval/distill.py`

구조 변경은 더 나중입니다.

- `src/lpcvc_retrieval/dual_tower.py`
- `src/lpcvc_retrieval/model.py`

## 자동 수정 허용 범위

현재 단계에서 자동 탐색 허용:

- `train.lr`
- `train.weight_decay`
- `train.warmup_epochs`
- `data.batch_size`
- `loss.w_contrastive`
- `loss.w_distill_affinity`
- `loss.w_hard_negative`
- `loss.w_text_text`
- `distill.affinity_temp`
- `distill.affinity_temp_start`
- `distill.affinity_temp_end`
- `distill.distill_margin_thr`
- `data.allowed_sources`

현재 단계에서 자동 수정 금지:

- `src/lpcvc_retrieval/data.py`
- `src/lpcvc_retrieval/metrics.py`
- `src/lpcvc_retrieval/config.py`
- `scripts/eval.py`
- `src/lpcvc_retrieval/export.py`
- `compile_and_profile.py`

## 평가 기준

현재 실험 루프의 주 메트릭:

- `val_mean_r10 = (I2T_R@10 + T2I_R@10) / 2`

같이 기록할 것:

- `I2T_R@10`
- `T2I_R@10`
- `train/loss`
- `train/distill_loss`
- crash / OOM / NaN 여부

## Keep / Discard 기준

- 개선이 있고 학습이 안정적이면 `KEEP`
- 개선이 없거나 재현성이 낮으면 `DISCARD`
- crash / OOM / NaN이면 `CRASH`

주의:

- subset search 결과는 `방향 탐색용`입니다.
- 실제 채택은 full data 재검증 뒤에 확정합니다.

## 시간 예산 방식을 지금 쓰지 않는 이유

- 현재 학습 코드는 epoch 완주 후 평가 구조입니다.
- 시간을 강제로 끊는 방식은 `중간 종료 후 반드시 평가` 로직을 추가해야 안전합니다.
- 따라서 현재 레포에서는 `고정 subset + 적은 epoch 완주`가 더 안전합니다.

## 제거 가능 파일

아래 파일은 연구 보조용이며, 나중에 최종 제출 전에 제거하거나 무시할 수 있습니다.

- `program.md`
- `research/AUTORESEARCH_LITE_PLAN.md`
- `research/EXPERIMENTS.tsv`
- `scripts/preprocess/make_jsonl_subset.py`
- `configs/subset_*.yaml`

