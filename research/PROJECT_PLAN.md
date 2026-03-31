# Project Plan

작성일: 2026-03-22  
대상 프로젝트: `distillation_ver1`

---

## 1. 프로젝트 목표

이 프로젝트의 목표는 `모바일 제약을 고려한 image-text retrieval 모델`을 만드는 것입니다.

현재 기본 방향은 아래와 같습니다.

- student: `MobileNetV4 Hybrid Large` + `DatologyAI/retr-opt-vit-b-32`
- teacher: `ViT-gopt-16-SigLIP2-256` single teacher
- baseline data: `coco + flickr30k`
- 기본 원칙: 복잡한 아이디어를 많이 넣기보다, `해석 가능한 baseline`을 먼저 확보

---

## 2. 현재 기본 전략

### 유지할 것

- single teacher 전략
- `coco + flickr30k` baseline
- no-teacher -> single-teacher -> baseline 순서
- 자세한 로깅과 실험 기록
- 한 번에 한 가지 큰 변화만 검증

### 나중에 확장할 것

- `open_images` filtered 비교
- hard negative ablation
- text-text ablation
- offline feature mode
- JEST/SIEVE식 data subset selection

### 지금 하지 않을 것

- dual teacher adaptive 주력화
- `wit`를 baseline에 바로 포함
- 객체 단위 평가로 문제 변경
- 여러 실험 변수를 동시에 섞기
- full autoresearch self-modifying loop 도입

---

## 3. 현재 체크해야 할 핵심 질문

### 시스템 질문

- config가 정상 로드되는가
- 데이터 경로가 맞는가
- teacher 로딩이 되는가
- eval과 checkpoint 저장이 되는가
- NaN 없이 끝나는가

### 학습 질문

- train loss가 줄어드는가
- distill loss가 줄어드는가
- validation I2T/T2I가 함께 좋아지는가
- baseline과 비교할 수 있는가

### 데이터 질문

- `coco + flickr30k` baseline이 안정적인가
- `open_images`를 넣으면 좋아지는가, 아니면 노이즈만 늘어나는가
- source별 caption 스타일 차이가 결과를 흔드는가

### 배포 질문

- ONNX export가 유지되는가
- compile/profile 경로가 깨지지 않는가

---

## 4. 단계별 실행 순서

### Phase 1. Smoke

1. `no_teacher_smoke`
2. `single_teacher_smoke`

성공 기준:

- train loop 정상
- eval 정상
- checkpoint 저장
- NaN 없음
- single teacher smoke에서 distill loss 계산 확인

### Phase 2. Baseline

1. `single_teacher_baseline`
2. 결과표 작성
3. keep/discard 판단

성공 기준:

- 반복 가능한 결과
- validation metric 해석 가능
- 이후 ablation의 기준점으로 사용 가능

### Phase 3. Data Expansion

1. `open_images` 추가 버전
2. `open_images` 길이/품질 필터 버전
3. baseline과 비교

성공 기준:

- baseline보다 의미 있는 개선
- 왜 좋아졌는지 설명 가능

### Phase 4. Loss Ablation

1. hard negative on/off
2. text-text on/off

성공 기준:

- baseline 대비 개선
- training instability 없음

### Phase 5. Deployment Check

1. best checkpoint 정리
2. ONNX export
3. compile/profile

성공 기준:

- accuracy와 배포 경로가 함께 유지

---

## 5. 실험 운영 원칙

- baseline 없이 고급 튜닝부터 시작하지 않는다.
- 한 번에 한 가지 큰 변화만 본다.
- 실험 전 가설을 적는다.
- 실험 후 keep/discard를 기록한다.
- 결과가 좋더라도 왜 좋았는지 설명할 수 있어야 한다.

---

## 6. 다음 행동

현재 가장 먼저 할 일:

1. `research/EXPERIMENTS.tsv` 기준으로 smoke 실험 등록
2. `no_teacher_smoke` 실행
3. 기록 업데이트
4. `single_teacher_smoke` 실행
5. 기록 업데이트
