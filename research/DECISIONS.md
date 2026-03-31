# Decisions

작성일: 2026-03-22

이 문서는 프로젝트에서 왜 어떤 선택을 했는지 남기는 기록입니다.

---

## Decision 001. 기본 전략은 dual teacher가 아니라 single teacher로 간다

날짜: 2026-03-22  
상태: 채택

### 결정

- 기본 teacher는 `ViT-gopt-16-SigLIP2-256` single teacher를 사용한다.

### 이유

- dual teacher adaptive는 해석이 어렵다.
- baseline 없이 복잡한 teacher mixing을 먼저 다루기 어렵다.
- 일정상 빠르게 설명 가능한 기준점을 만드는 것이 더 중요하다.

### 영향

- `config.yaml` 기본값이 single teacher baseline으로 변경됨
- dual teacher는 legacy comparison config로만 유지

---

## Decision 002. baseline 데이터는 `coco + flickr30k`로 시작한다

날짜: 2026-03-22  
상태: 채택

### 결정

- 초기 baseline에서는 `coco + flickr30k`만 사용한다.

### 이유

- retrieval supervision 구조와 잘 맞는다.
- caption 스타일이 상대적으로 안정적이다.
- 결과 해석이 쉽다.

### 영향

- `config.yaml`에 `allowed_sources` 추가
- `data.py`에 source filtering 기능 추가

---

## Decision 003. `open_images`는 나중에 filtered ablation으로 본다

날짜: 2026-03-22  
상태: 채택

### 결정

- `open_images`는 baseline에 바로 넣지 않고, baseline 이후 필터링 버전으로 비교한다.

### 이유

- 장문 caption과 노이즈가 많다.
- baseline 해석을 흐릴 수 있다.

### 영향

- baseline 성공 후 data expansion phase에서 다룸

---

## Decision 004. `wit`는 초기 baseline에서 제외한다

날짜: 2026-03-22  
상태: 채택

### 결정

- `wit`는 baseline 단계에서 사용하지 않는다.

### 이유

- 다국어와 제목형 문장이 많아 현재 retrieval baseline과 결이 다를 수 있다.
- 이득보다 해석 혼란이 더 클 수 있다.

### 영향

- 필요하면 마지막에 별도 ablation으로만 확인

---

## Decision 005. baseline에서는 extra loss를 끈다

날짜: 2026-03-22  
상태: 채택

### 결정

- baseline에서는 `w_rank`, `w_hard_negative`, `w_text_text`를 0으로 둔다.

### 이유

- teacher 효과와 extra loss 효과를 분리하고 싶다.
- baseline은 기준점이어야지 복합 실험이 되어선 안 된다.

### 영향

- hard negative, text-text는 baseline 이후 ablation으로 재검토

---

## Decision 006. `autoresearch`는 full 도입이 아니라 운영 원칙만 차용한다

날짜: 2026-03-22  
상태: 채택

### 결정

- full self-modifying autoresearch loop는 도입하지 않는다.
- 대신 baseline-first, one-change-at-a-time, keep/discard, results log 원칙만 가져온다.

### 이유

- 현재 프로젝트는 다중 파일 구조이고 metric도 하나로 단순하지 않다.
- 지금은 자동화보다 사람이 실험 의미를 이해하는 것이 더 중요하다.

### 영향

- `research/EXPERIMENTS.tsv`와 `research/templates/experiment_template.md`를 기준으로 반자동 운영

---

## Decision 007. 현재 레포에는 `autoresearch-lite`를 적용한다

날짜: 2026-03-24  
상태: 채택

### 결정

- full autonomous code rewriting은 지금 도입하지 않는다.
- `config-only autoresearch`를 먼저 적용한다.
- `고정 subset + 적은 epoch 완주` 전략을 사용한다.
- 배포 검증은 최종 후보가 정해진 뒤에만 수행한다.
- 코드 수정은 사람 승인 후에만 진행한다.

### 이유

- 현재 레포는 다중 파일 구조이고 epoch 끝 평가 구조다.
- time-budget 방식은 추가 안전 장치가 없으면 비교가 불안정해질 수 있다.
- 로컬 RTX 2060 환경에서는 full data 반복 탐색 비용이 높다.
- config 탐색은 리스크가 낮고 제거도 쉽다.

### 영향

- `program.md`를 기준 지시서로 사용
- subset JSONL과 subset config를 사용한 실험 루프 도입
- 연구용 보조 파일은 나중에 최종 제출 전에 제거 가능
