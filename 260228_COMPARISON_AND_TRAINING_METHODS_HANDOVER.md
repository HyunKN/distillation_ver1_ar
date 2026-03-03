# 프로젝트 인수인계서: 비교 결과 + 학습 기법 확장안

작성일: 2026-02-28  
목적: 다음 작업자가 바로 이어서 실험/개발을 진행할 수 있도록 현재 판단과 다음 액션을 정리

---

## 1) 이번 문서의 전제
- 두 프로젝트 비교는 **모델 자체 성능 차이는 제외**하고, 학습 코드 방식/효율/확장성만 평가함.
- 데이터 파서/경로 이슈는 제외하고 학습 파이프라인(`train.py`, `distill.py`, `losses.py`, `config.yaml`) 중심으로 판단함.

---

## 2) 비교 대상
1. `mjProject`
2. `MobileCLIP2-Retrieval-Optimization` (현재 메인)

핵심 비교 보고서:
- `MJPROJECT_VS_MOBILECLIP2_REPORT_2026-02-28.md`

---

## 3) 비교 결론 (요약)
선정: **`MobileCLIP2-Retrieval-Optimization` 유지**

선정 근거:
1. 학습 기법 확장성(옵션 on/off 및 ablation 축)이 더 넓음
2. 듀얼 teacher, 추가 loss, EMA, compile 등 성능 레버가 코드에 이미 반영됨
3. `mjProject`의 단순 베이스라인 전략은 현재 프로젝트에서 부분 재현 가능

단, 사실:
- 순수 속도(throughput)만 보면 `mjProject`가 유리할 가능성이 큼(단일 teacher + loss 단순).

---

## 4) 코드 기준 팩트체크 포인트

### 4.1 공통 학습 골격
두 프로젝트 공통:
1. `AdamW`
2. cosine warmup scheduler
3. AMP(`autocast` + `GradScaler`)
4. gradient clipping

근거:
- `mjProject/src/lpcvc_retrieval/train.py:194`
- `mjProject/src/lpcvc_retrieval/train.py:198`
- `mjProject/src/lpcvc_retrieval/train.py:207`
- `mjProject/src/lpcvc_retrieval/train.py:310`
- `src/lpcvc_retrieval/train.py:217`
- `src/lpcvc_retrieval/train.py:221`
- `src/lpcvc_retrieval/train.py:230`
- `src/lpcvc_retrieval/train.py:363`

### 4.2 차이점
1. Teacher 구조
- `mjProject`: 단일 teacher (`mjProject/src/lpcvc_retrieval/distill.py:38`)
- 현재 프로젝트: 다중 teacher 순회 + 가중합 (`src/lpcvc_retrieval/distill.py:171`, `src/lpcvc_retrieval/distill.py:193`, `src/lpcvc_retrieval/distill.py:265`)

2. 추가 loss
- `mjProject`: 기본 contrastive + ranking 중심
- 현재 프로젝트: hard-negative + text-text 추가
- 근거: `src/lpcvc_retrieval/train.py:315`, `src/lpcvc_retrieval/train.py:323`

3. 안정화/효율 옵션
- 현재 프로젝트: `torch.compile`, EMA, `persistent_workers` 사용
- 근거: `src/lpcvc_retrieval/train.py:152`, `src/lpcvc_retrieval/train.py:272`, `src/lpcvc_retrieval/train.py:178`

---

## 5) 초격차 성능을 위한 최신 Distillation 방법론 (A100 환경 최적화)

기존 2023~2024년 방식(단순 Contrastive Loss, ALBEF/BLIP 등)을 넘어, 최근 학계(2025~2026)에서 집중하고 있는 **"어떻게 거대 VLM의 지식을 작은 학생 모델에 손실 없이(효율적으로) 구겨 넣을 것인가"**에 대한 최신 트렌드를 현재 우리 코드에 접목할 수 있도록 정리했습니다.

1. **초대규모 배치를 위한 Offline Feature Extraction (사전 임베딩 추출)**  
   - **원리**: A100의 넉넉한 VRAM을 100% 활용하기 위해, 부피가 큰 Teacher 모델들(SigLIP Giant, PE-Core bigG/14 등)을 메모리에 상주시키지 않고 사전에 정답(Embedding Logit)만 `.pt` 파일로 추출해둡니다.  
   - **기대효과**: VRAM을 오로지 학생 모델과 Batch Size에 전부 몰아줄 수 있게 됩니다. Dual Teacher 학습 과정에서도 Contrastive Loss의 핵심인 '초거대 배치 사이즈(Macro Batch)'를 달성하여 검색(Retrieval) 성능의 한계를 돌파하는 프레임워크 뼈대입니다.

2. **DCLIP: Cross-Modal Transformer Distillation (2025 최신 검색 목적 증류)**  
   - **관련 연구**: Distill CLIP (DCLIP): Enhancing Image-Text Retrieval via Cross-Modal Transformer Distillation (2025. 05)
   - **원리**: 단순한 글로벌 벡터 매칭(ITC)을 넘어, Teacher의 지역적(Regional) 특징까지 학생에게 강제 매핑합니다. 
   - **목적/이득**: Zero-shot 분류 성능은 유지하면서, 이미지-텍스트 검색(Retrieval)이라는 우리 프로젝트의 본질적 목표(R@1)를 가장 직접적으로 수직 상승시켜주는 최신 접근법입니다.
   - **참고 수치**: 유사 환경(ViT-B Student)에서 Retrieval R@1이 기존 대비 **+2~4%p** 상승 보고.
   - **⚡ CFD-CLIP과의 차이**: DCLIP은 이미지 내부의 **"어디(Where)"**에 집중합니다. 객체 검출(YOLO 등) 기반으로 이미지의 특정 영역(Region)과 텍스트 토큰을 1:1로 대응시키는 **지역적(Local) 매핑**입니다. → 객체가 여러 개 등장하는 복잡한 이미지 검색에 강합니다.

3. **CFD-CLIP: Contrastive Feature Distillation (2025 듀얼 대조 학습)**  
   - **관련 연구**: CFD-CLIP: Contrastive Feature Distillation with CLIP for Image Classification (2025. 09)
   - **원리**: 학생 모델이 단순히 Teacher의 출력 '결괏값(Logit)'만 따라 하는 것이 아니라, Teacher의 '임베딩 공간(Embedding Space)' 그 자체의 구조를 모방하도록 이중(Dual-contrastive) 손실 함수를 설계합니다.
   - **목적/이득**: 단순히 정답을 외우는 것을 넘어, Teacher가 세상을 바라보는 기준 체계를 이식받아 모델의 근본적인 일반화 성능이 비약적으로 강화됩니다.
   - **참고 수치**: Classification 기준 Teacher 대비 성능 유지율 **90~95%** 달성 보고 (기존 Logit-only 방식은 ~85%).
   - **⚡ DCLIP과의 차이**: CFD-CLIP은 **"어떻게(How)"** 세상을 바라보는지에 집중합니다. 이미지 안의 특정 영역은 신경 쓰지 않고, Teacher의 **전체적인 임베딩 분포 구조(글로벌 기하학)**를 학생에게 통째로 이식합니다. → 처음 보는 데이터(Unseen Domain)에 대한 일반화 능력에 강합니다.
   - **💡 실전 도입 기준**: 두 기법은 상호 배타적이지 않습니다. **DCLIP(지역 매핑) + CFD-CLIP(공간 구조 이식)을 동시에 적용**하면 지역적 정밀도와 전역적 일반화를 모두 챙길 수 있으며, 이것이 가장 이상적인 조합입니다.

4. **AMMKD: Adaptive Multimodal Multi-teacher Distillation (2025 듀얼 티처 종결자)**  
   - **관련 연구**: AMMKD (2025. 08 출판) / TinyCLIP (2023)의 진화형
   - **원리**: 우리 프로젝트의 핵심인 **Dual Teacher 구조에 직접적으로 꽂히는 최신 논문**입니다. 고정된 비율(0.5 : 0.5)로 Teacher를 섞는 것이 아니라, 데이터 쌍마다 어떤 Teacher의 지식이 더 정답에 가까운지 네트워크가 동적으로 가중치(Adaptive Weight)를 조절하고, 데이터 간의 관계성(Affinity Matrix)를 증류합니다.
   - **목적/이득**: SigLIP Giant가 잘하는 분야와 PE-Core bigG/14가 잘하는 분야를 학생 모델이 스위칭해가며 뷔페처럼 떠먹게 만들어, 듀얼 티처의 잠재력을 극대화합니다.
   - **참고 수치**: Multi-teacher 환경에서 고정 가중치 대비 Adaptive 방식이 **+1~3%p** 우위 보고 (TinyCLIP 계열 벤치마크 기준).

---

## 6) 추가 연구 및 발전 (대회 후반 / 논문 고도화용)

1. **PAND: Prompt-Aware Neighborhood Distillation (2026. 02 최신)**  
   - **원리**: 기존 증류 기법이 모델의 전체적인 방향(Global Alignment)만 맞추려다 세밀한 관계를 놓치는 점을 지적합니다. PAND는 글로벌 의미 교정(Semantic Calibration)과 세밀한 구조 전이(Structural Transfer)를 분리(Decoupling)하여 처리합니다.
   - **적용 의미**: 데이터가 1만 장으로 적은 우리 프로젝트에서, 단순한 거시적 학습이 끝나고 "고양이를 쫓는 개" vs "개를 쫓는 고양이" 같은 미세한(Fine-grained) 의미망을 학생 모델에게 각인시킬 때 쓰일 궁극의 무기입니다.

2. **LAid: Long-window Anchoring in VLM Distillation (2025. 12)**  
   - **의미**: 캡션의 길이가 아주 길어지면 작은 학생 모델이 문맥을 잃어버리는 현상을 보완합니다. 텍스트 길이가 극단적으로 긴 캡션 데이터가 섞여 있다면 도입해볼 만한 25년식 해결책입니다.

---

## 7) 적용 우선순위 및 전략 (A100 & 2026 기준 실전 로드맵)

기본 원칙:
1. 한 번에 하나만 켜서 원인-결과를 분리한다.
2. ROI가 큰 순서로 적용한다.
3. 현재 코드 구조(Offline Distill + EMA + SigLIP)를 깨지 않는 범위에서 확장한다.

### 7.1 확정 운영 방침
1. `SWA`는 현재 단계에서 제외한다.
2. `강한 Multi-crop`은 현재 단계에서 제외한다.
3. `EMA`는 유지한다.
4. 최우선은 `Temperature Scheduling`이다.

제외 사유:
1. SWA는 EMA와 역할이 겹쳐 ablation 해석이 흐려진다.
2. 강한 Multi-crop은 현재 Offline Teacher 타깃과의 의미 괴리를 키워 distill noise를 유발할 수 있다.

### 7.2 최적 적용 순서 (실행 기준)
1. **Step 0: Offline 파이프라인 고정**
- 목표: 재현 가능한 baseline 확보
- 작업: `extract_features.py` 실행 시 `data.train_augment=false`로 Teacher feature를 고정 추출
- 작업: `run_train.py` 학습 시 `data.train_augment=true` 유지
- 기대효과: Teacher=고정(Global), Student=증강(일반화) 비대칭 구조를 안정적으로 확보

2. **Step 1: Distillation Temperature Scheduling**
- 목표: 초반 불안정 완화 + 후반 분별력 강화
- 작업: `distill.affinity_temp`를 고정값 대신 epoch 기반 스케줄로 변경
- 권장: `temp_start`(soft) -> `temp_end`(sharp), cosine 혹은 linear
- 기대효과: `R@1 +0.2 ~ +0.8%p` (환경 의존)

3. **Step 2: Adaptive Teacher Weight (Offline 호환 방식)**
- 목표: 고정 0.5/0.5의 한계 해소
- 작업: teacher별 배치 품질 점수(`pos_sim - neg_sim margin`)로 동적 가중치 계산
- 작업: `w = softmax(margin/tau)` + `w_min` 바닥값 적용
- 기대효과: `R@1 +0.5 ~ +1.5%p` (환경 의존)

쉬운 설명:
1. 기존 방식은 모든 배치에서 항상 `0.5 : 0.5`로 teacher를 섞습니다.
2. Adaptive 방식은 배치가 바뀔 때마다 "이번 배치에서 누가 더 잘 가르쳤는지"를 계산합니다.
3. 더 잘 가르친 teacher 비중을 높이고, 덜 맞는 teacher 비중을 낮춰서 distill loss를 만듭니다.
4. 즉, teacher 1명만 고르는 것이 아니라 두 teacher를 상황에 맞게 비율로 섞습니다.

판단 기준(현재 구현):
1. 각 teacher에 대해 유사도 행렬 `S_t = img_emb_t @ txt_emb_t^T` 계산
2. `image_id` 기준으로 positive/negative 쌍 분리
3. 점수 계산:
   - `score_t = mean(S_t[positive]) - mean(S_t[negative])`
4. 가중치 변환:
   - `w = softmax(score / tau)`
   - `w_min`으로 한 teacher 쏠림(collapse) 방지

배치 단위 예시:
1. 배치 A: `score1=0.38`, `score2=0.22` -> `w1=0.74`, `w2=0.26` (예시)
2. 배치 B: `score1=0.15`, `score2=0.19` -> `w1=0.44`, `w2=0.56` (예시)
3. 따라서 비율은 고정이 아니라 매 배치마다 달라집니다.

4. **Step 3: False-Negative 보정**
- 목표: 실제로 유사한 샘플을 negative로 과벌점하는 문제 완화
- 작업: image_id 외의 유사 샘플에 대한 negative 완화 규칙 추가
- 기대효과: `R@1 +0.3 ~ +1.0%p` (환경 의존)

5. **Step 4: 고난도 기법(CFD/PAND 등) 검토**
- 목표: 논문/졸업 연구용 고도화
- 조건: Step 1~3이 안정적으로 끝난 뒤에만 진행
- 주의: 구현량 대비 대회 일정 ROI가 낮을 수 있음

### 7.3 왜 이 순서가 가장 효율적인가
1. Step 1은 코드 변경이 작고 리스크가 가장 낮다.
2. Step 2는 효과가 크지만 기준선(temperature)이 먼저 안정되어야 해석이 가능하다.
3. Step 3은 데이터 분포 영향이 커서 앞 단계가 정리된 뒤 적용해야 안전하다.

### 7.4 전체 후보 포함 구현 순서 (이번에 조사한 기법 전부 반영)

아래 순서는 "현재 코드 + 조사한 기법 전체"를 한 로드맵으로 합친 최종 버전이다.

| 순서 | 기법 | 상태 | 구현 우선순위 | 비고 |
|---|---|---|---|---|
| 0 | Offline 파이프라인 고정(추출 OFF/학습 ON) | 적용됨 | 필수 | 기준선 재현성 확보 |
| 1 | Distillation Temperature Scheduling | 적용됨 | 필수 | Step 1 완료 |
| 2 | Adaptive Teacher Weight (offline margin 기반) | 적용됨(기본 OFF) | 즉시(A/B) | 코드 구현 완료, `distill.adaptive_teacher_weight=true`로 활성화 |
| 3 | False-Negative 보정 (Debiased 계열) | 미적용 | 즉시 | 유사 샘플 과벌점 완화 |
| 4 | Hard-Negative/Text-Text 가중치 재튜닝 | 부분적용 | 즉시 | 기능은 있음, 최적값 미확정 |
| 5 | Selective distill 강화 (margin/columns/schedule) | 부분적용 | 중간 | 현재 selective는 있음, 스케줄은 미정 |
| 6 | EMA warm-start 스케줄(초반 off 후 on) | 미적용 | 중간 | EMA는 이미 있음 |
| 7 | SWA 단독 ablation (EMA off 조건) | 보류 | 조건부 | EMA와 중복, 비교 목적일 때만 |
| 8 | CFD-CLIP (feature geometry distill) | 미적용 | 연구 | 구현량 큼, 논문 트랙 |
| 9 | DCLIP (cross-modal transformer distill) | 미적용 | 연구 | 구조 침습 큼, 대회 ROI 낮음 |
| 10 | PAND / LAid | 미적용 | 연구 | 후반 고도화 전용 |
| 11 | 강한 Multi-crop | 보류 | 조건부 | 현재 offline 구조와 충돌 위험 |

운영 기준:
1. `필수/즉시` 트랙(0~4) 완주 후에만 `중간/조건부/연구`로 이동한다.
2. 각 단계는 단독 ablation으로 검증한다.
3. 다음 단계로 넘어가는 기준은 `R@1` 개선 + 변동폭 감소 + 학습시간 허용 범위 충족이다.

---

## 8) 즉시 실행 가능한 실험 템플릿 (현행 합의안 반영)

실험 로그 필수 지표:
1. `R@1, R@5, R@10` (I2T/T2I)
2. `time/epoch`
3. 최대 VRAM(GB)
4. `distill loss`, `temperature`, teacher weight 로그

실험군:

| 실험 ID | 적용 기법 | 목적 | 핵심 관찰 포인트 |
|---------|----------|------|------------------|
| Exp-0 | Baseline (현재 코드) | 기준선 확립 | R@K, VRAM, time/epoch |
| Exp-1 | Exp-0 + Offline 운영 규칙 고정(추출 OFF/학습 ON) | 재현성 + 안정화 | 재실행 간 성능 분산 감소 |
| Exp-2 | Exp-1 + Temperature Scheduling | 기준선 안정화 | 초반 수렴 안정성, 후반 R@1 |
| Exp-3 | Exp-2 + Adaptive Teacher Weight | 듀얼 teacher 효율화 | 고정 0.5/0.5 대비 개선폭 |
| Exp-4 | Exp-3 + False-Negative 보정 | hard case 일반화 | R@1/R@5 개선, 오탐 감소 |
| Exp-5 | Exp-4 + Hard-Negative/Text-Text 재튜닝 | loss 밸런싱 최적화 | distill/main loss 균형 |
| Exp-6 | Exp-5 + EMA warm-start | 안정성 향상 | val 곡선 진동폭 감소 |
| Exp-7 | (조건부) EMA OFF + SWA ON 단독 비교 | EMA/SWA 중복 해소 검증 | 동일 예산에서 최종 R@1 비교 |
| Exp-8 | (연구) CFD-CLIP | 임베딩 구조 이식 | 일반화 성능 변화 |
| Exp-9 | (연구) DCLIP | 지역-텍스트 정렬 강화 | 복잡도 대비 ROI |
| Exp-10 | (연구) PAND/LAid | 미세 의미 정렬 | 장기 문맥/미세 구문 성능 |

실행 규칙:
1. 실험당 변경점은 1개만 허용
2. 2회 이상 반복 측정 후 평균/표준편차 기록
3. 향상폭이 작아도 변동폭이 줄면 채택 후보로 유지

### 8.1 Adaptive Teacher Weight A/B 실행 절차 (필수)

비교 목적:
1. 고정 teacher 비율(`0.5/0.5`) 대비 adaptive 동적 가중합이 실제로 이득인지 확인
2. "느낌"이 아니라 동일 조건 수치로 채택 여부 결정

실행 방법:
1. **A (고정 가중치)**  
   - `distill.adaptive_teacher_weight: false`
2. **B (동적 가중치)**  
   - `distill.adaptive_teacher_weight: true`
3. A/B 모두 아래 조건을 동일하게 유지  
   - 동일 데이터셋, 동일 seed, 동일 epoch, 동일 batch, 동일 checkpoint 초기값

판정 지표:
1. `R@1`을 1순위 지표로 본다
2. `R@5`, `R@10`, `time/epoch`, `val 변동폭`을 함께 본다
3. 2회 이상 반복 평균으로 결정한다

채택 기준(권장):
1. `R@1`이 유의미하게 상승(또는 동일 성능에서 변동폭 감소)하면 adaptive ON 채택
2. 이득이 불명확하면 고정 가중치 유지

운영 권장:
1. 대회 실전 운영은 **adaptive ON 우선**이 합리적이다
2. 단, 첫 1~2회 A/B 확인 없이 영구 ON 고정은 권장하지 않는다

### 8.2 A100 서버 첫 실행 권장 범위

처음 서버에서 돌릴 때는 아래 순서로 진행한다.

1. **첫 실행(안전 모드): Step 0 + Step 1까지만 적용**
- 적용: Offline 파이프라인 + Temperature Scheduling
- 비적용: Adaptive Teacher Weight(`adaptive_teacher_weight=false`)
- 목적: 환경 문제/데이터 경로/추출 feature 정합성부터 안정 확인

2. **두 번째 실행(성능 모드): Step 2까지 적용**
- 적용: 위 설정 + Adaptive Teacher Weight(`adaptive_teacher_weight=true`)
- 목적: 고정 0.5/0.5 대비 실제 개선폭 확인

3. **처음부터 켜지 말 것**
- False-Negative 보정, 연구 트랙(CFD/DCLIP/PAND/LAid)은 첫 A100 실행에서 제외

첫 실행 시 체크포인트:
1. 추출 로그: `Effective data.train_augment: False`
2. 학습 로그: `Offline mode — Teacher NOT loaded.`
3. 학습 로그: `distill_temp_schedule=...`
4. (성능 모드) 학습 로그: `adaptive_teacher_weight=True`

---

## 9) 다음 작업자 TODO 체크리스트 (우선순위 반영)
- [x] Offline Feature Extraction 구현 완료 (11번 변경 이력)
- [x] caption_idx round-trip + dataset fingerprint 검증 완료
- [x] `config.yaml`에 distill temperature schedule 필드 추가 (`affinity_temp_start`, `affinity_temp_end`, `affinity_temp_schedule`)
- [x] `train.py`에서 epoch별 distill temperature 갱신 로직 추가
- [x] `distill.py`에 adaptive teacher weighting (offline margin 기반) 옵션 추가
- [ ] False-Negative 보정 loss 추가(또는 기존 loss에 debias 항 반영)
- [ ] Hard-Negative/Text-Text 가중치 sweep 범위 확정 (`w_hard_negative`, `w_text_text`)
- [ ] EMA warm-start 옵션 추가 (초기 N epoch EMA update skip)
- [ ] 실험 config 파일 분리 (`baseline`, `temp_sched`, `adaptive_weight`, `fn_correction`)
- [ ] 실험 결과 테이블 (`R@K`, 시간, VRAM, 분산) 갱신

보류 항목:
- [ ] SWA (EMA와 중복, 후순위)
- [ ] 강한 Multi-crop (현재 오프라인 구조와 충돌 위험, 후순위)

연구 트랙(졸업 논문/후반 확장):
- [ ] CFD-CLIP 구현 가능성 검토 및 PoC
- [ ] DCLIP 구현 가능성 검토 및 PoC
- [ ] PAND/LAid 도입 조건 정의 (데이터/예산/목표 지표)

---

## 10) 최종 메모 (Executive Summary)

핵심 3줄:
1. 현재 프로젝트는 이미 `SigLIP + Dual Teacher + Offline Distill + EMA` 기반의 강한 실전형 구조를 갖추고 있다.
2. 다음 성능 개선의 최단 경로는 `Temperature Scheduling -> Adaptive Teacher Weight -> False-Negative 보정` 순서다.
3. 조사한 전체 기법은 본문 7.4에 모두 포함했으며, `즉시 트랙(0~4)`과 `연구 트랙(8~10)`으로 분리 운영한다.

운영 원칙:
1. 추출 단계와 학습 단계의 증강 전략을 분리한다.
2. 실험은 한 번에 한 가지 변화만 적용한다.
3. 모든 의사결정은 `R@K + 시간 + VRAM + 변동폭` 4개 지표로 한다.

### 10.1 현재 기본 Teacher 조합 출처 (2026-03 기준)
1. Teacher 1: `ViT-gopt-16-SigLIP2-256` (`webli`)
   - GitHub: https://github.com/google-research/big_vision
   - Hugging Face: https://huggingface.co/timm/ViT-gopt-16-SigLIP2-256
   - 라이선스(모델 카드 표기): Apache-2.0
2. Teacher 2: `PE-Core-bigG-14-448` (`meta`)
   - GitHub: https://github.com/facebookresearch/perception_models
   - Hugging Face: https://huggingface.co/facebook/PE-Core-G14-448
   - OpenCLIP 매핑: https://huggingface.co/timm/PE-Core-bigG-14-448
   - 라이선스(모델 카드 표기): Apache-2.0
3. 로딩 프레임워크: OpenCLIP
   - GitHub: https://github.com/mlfoundations/open_clip
   - 라이선스: MIT

---

## 11) 변경 이력

### 2026-03-01: Offline Feature Extraction 구현 (Method B)

Teacher 모델(SigLIP Giant, PE-Core bigG/14)을 VRAM에 상주시키지 않고, 사전 추출된 임베딩을 DataLoader에서 공급하는 구조를 구현했습니다.

**변경된 파일:**

| 파일 | 변경 내용 |
|------|----------|
| `scripts/extract_features.py` | [NEW] Teacher 임베딩 사전 추출 스크립트 (Float16 저장) |
| `src/lpcvc_retrieval/data.py` | `OfflineFeatureDataset` Wrapper 추가, `collate_fn` 확장 |
| `src/lpcvc_retrieval/distill.py` | `get_teacher_output()` 통합 함수 추가, `DistillConfig`에 `offline_feature_dir` 필드 추가 |
| `src/lpcvc_retrieval/train.py` | 배치 언팩킹 수정 + `get_teacher_output()` 호출 |
| `config.yaml` | `distill.offline_feature_dir: null` 설정 추가 |

**검증 결과:**
- Mock 단위 테스트 15개 전부 통과
- End-to-End 스모크 테스트 (ViT-B-32 + COCO, CPU) 45+ 배치 에러 없이 작동 확인

**버그 수정 (코드 리뷰 후 패치):**

| 문제 | 수정 내용 |
|------|----------|
| `make_datasets`가 `val_ds`까지 래핑 → eval 루프 깨짐 | `train_ds`만 래핑하도록 수정 (`data.py`) |
| `extract_features.py` 재실행 시 자동 래핑 오염 | 추출 전 `offline_feature_dir=null` 강제 설정 |
| `use_teacher: false`여도 offline 임베딩 시 distill 실행 | `use_offline`을 `use_teacher` 게이트로 보호 (`train.py`) |
| feature 무결성 체크가 길이만 검증 | img/txt 차원 불일치 검증 추가 (`data.py`) |
| 추출/학습 간 캡션 랜덤 선택 불일치 → distill noise | `caption_idx` round-trip 구현 (아래 참조) |
| 추출 후 데이터셋 순서/내용 변경 시 조용히 오학습 | dataset fingerprint 실시간 대조 추가 (`data.py`) |

**캡션 인덱스 동기화 (caption_idx round-trip):**

추출 시점과 학습 시점에 같은 캡션을 사용하도록 보장하는 메커니즘:

```
Dataset.__getitem__(idx) → meta["caption_idx"]에 선택된 캡션 번호 기록
  → extract_features.py가 caption_indices를 .pt에 저장
    → OfflineFeatureDataset이 .pt에서 읽어 set_forced_caption_indices()로 주입
      → Dataset.__getitem__이 forced[idx]로 추출 시점과 동일한 캡션 선택
```

`.pt` 파일에 저장되는 무결성 정보:
- `caption_indices`: 추출 시 사용된 캡션 인덱스 배열 `[N]`
- `sample_count`: 샘플 수 (이중 검증)
- `dataset_fingerprint`: SHA1(image_id|img_rel) — 데이터셋 순서 변경 감지

**데이터셋 fingerprint 실시간 대조:**

`OfflineFeatureDataset`은 초기화 시 현재 Dataset의 `samples`로부터 fingerprint를 재계산하여 `.pt`에 저장된 값과 비교합니다. 추출 이후 JSONL 순서 변경, 샘플 추가/삭제 등이 발생하면 즉시 `ValueError`를 발생시켜 잘못된 임베딩으로 학습하는 것을 방지합니다. 사용자가 별도로 설정할 것은 없으며 완전 자동으로 작동합니다.

```
[추출 시] extract_features.py
  Dataset.samples 순회 → SHA1("{image_id}|{img_rel}\n") 누적 → .pt에 저장

[학습 시] OfflineFeatureDataset.__init__
  현재 Dataset.samples로 동일 방식 SHA1 재계산
  → .pt의 fingerprint와 비교
  → 일치하면 통과, 불일치하면 즉시 ValueError
```

정상 케이스:
```bash
python scripts/extract_features.py --config config.yaml --out_dir features/ --override data.train_augment=false
# config.yaml에서 data.train_augment: true 유지
python run_train.py   # fingerprint 일치 → 정상 학습 ✅
```

데이터 변경 시 자동 차단:
```bash
# train.jsonl에 새 데이터 추가 또는 순서 변경 후...
python run_train.py
# → ValueError: dataset_fingerprint mismatch between current dataset
#   and feature file teacher_0_train.pt.
#   Re-run scripts/extract_features.py.  ❌
```

**feature 재추출이 필요한 경우:**
- JSONL 파일의 순서나 내용이 변경된 경우 (fingerprint mismatch 에러 발생)
- Teacher 모델을 변경한 경우
- 캡션 매핑을 바꾸고 싶은 경우 (새로운 랜덤 캡션 조합)

**사용법 (권장 운영 규칙 반영):**
```bash
# 1단계: Teacher 임베딩 사전 추출 (추출 단계는 증강 OFF 권장)
python scripts/extract_features.py --config config.yaml --out_dir features/ --override data.train_augment=false

# 2단계: config.yaml에서 offline_feature_dir 설정
# distill:
#   offline_feature_dir: features/

# 3단계: 학습 실행 (학습 단계는 증강 ON 유지 권장)
# data.train_augment: true
python run_train.py --config config.yaml
```

**왜 이렇게 나누는가 (쉽게 설명):**
1. 추출 단계 OFF: Teacher 정답을 흔들리지 않는 기준점으로 고정
2. 학습 단계 ON: Student가 다양한 뷰를 보며 일반화 성능 확보
3. 결과: Teacher=안정적 기준, Student=강한 적응력을 동시에 달성

**하위 호환성:** `offline_feature_dir: null`이면 기존 Online 모드와 100% 동일하게 작동.

**핵심 API:**

- `get_teacher_output(teacher, imgs, metas, offline_teacher_embs, device)`
  - Online/Offline을 통합하는 단일 인터페이스. `train.py`는 이 함수만 호출하면 되며, 임베딩 출처를 알 필요 없음
  - `offline_teacher_embs`가 있으면 → 사전 추출 임베딩 사용 (Teacher Forward 생략)
  - `teacher`가 있으면 → 실시간 Forward 실행
  - 둘 다 없으면 → `None` 반환
  - 반환 형식: `(img_emb, txt_emb)` 또는 `[(img_emb, txt_emb), ...]` (Single/Multi Teacher)

- `OfflineFeatureDataset(base_dataset, feature_dir, split)`
  - 기존 Dataset을 감싸는 Wrapper. `teacher_{idx}_{split}.pt` 파일에서 임베딩을 로드하여 `(img, tokens, meta, teacher_embs)` 형태로 반환
  - 초기화 시 5단계 검증: 데이터셋 크기 → img/txt 차원 → `sample_count` → `dataset_fingerprint` (teacher 간 + 현재셋 대조) → `caption_indices` 교차 teacher 일치
  - train split에서 `caption_indices`가 있으면 자동으로 `set_forced_caption_indices()` 호출

- `set_forced_caption_indices(caption_indices)` (`JsonlRetrievalDataset`, `CocoCaptionsRetrievalDataset`)
  - 오프라인 모드에서 캡션 선택을 결정적으로 고정. `None` 전달 시 해제 (랜덤 복원)
