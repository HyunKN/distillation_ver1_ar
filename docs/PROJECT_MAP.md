# 프로젝트 길잡이 문서

이 문서는 현재 프로젝트의 지도입니다.  
계속 수정하다가 "지금 내가 뭘 하고 있었지?", "어떤 방향으로 프로젝트를 굴리고 있었지?"를 다시 빠르게 떠올리기 위한 복기용 문서입니다.

이 문서를 먼저 읽고, 더 자세한 구현은 `README.md`, `docs/PROJECT_GUIDE.md` 순서로 내려가면 됩니다.

주의:

1. 이 문서에 적힌 모델 조합은 영구 고정안이 아니라 `현재 기준 기본값`입니다.
2. teacher/student 조합은 실험 결과, 대회 조건, 라이선스, 서버 자원에 따라 언제든 바뀔 수 있습니다.
3. 따라서 이 문서는 "현재 어디까지 왔는가"를 복기하는 문서이지, "앞으로도 반드시 이 모델만 써야 한다"는 선언문이 아닙니다.

---

## 1. 이 프로젝트를 한 문장으로

이 프로젝트는 현재 `MobileNetV4 Hybrid Large` 이미지 인코더와 `DatologyAI/retr-opt-vit-b-32` 텍스트 인코더를 결합한 dual-tower 학생 모델을 기본값으로 두고, 대형 vision-language teacher의 지식을 증류해 모바일 환경용 이미지-텍스트 검색 성능을 높이려는 LPCVC 2026용 retrieval 프로젝트입니다.

단, 학생 모델 역시 실험 결과, 라이선스, 대회 조건에 따라 변경될 수 있습니다.

```text
┌────────────────────────────────────────────────────────────┐
│                        Teacher Models                      │
│  ┌────────────────────────┐   ┌─────────────────────────┐  │
│  │ ViT-gopt-16-SigLIP2-256│   │  PE-Core-bigG-14-448   │  │
│  │     (text-image align) │   │ (visual robustness)    │  │
│  └────────────┬───────────┘   └────────────┬────────────┘  │
│               │                            │               │
│               └──── adaptive teacher weighting ───────────┘│
│                                │                           │
└────────────────────────────────┼───────────────────────────┘
                                 ▼
                  ┌──────────────────────────────────────────┐
                  │        Dual-Tower Student (current)      │
                  │  Image: MobileNetV4 Hybrid Large         │
                  │  Text : DatologyAI/retr-opt-vit-b-32     │
                  │  contrastive + distill losses            │
                  └──────────────────────────────────────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
            Retrieval Evaluation          ONNX Export
           (I2T / T2I Recall@K)        + QAI Hub Compile/Profile
```

---

## 2. 현재 프로젝트의 핵심 목표

1. 모바일에서 돌릴 수 있는 경량 이미지-텍스트 retrieval 모델을 지향한다.
2. 현재 기준 학생 모델은 `MobileNetV4 Hybrid Large + DatologyAI/retr-opt-vit-b-32` dual-tower 구조로 본다.
3. 현재 운영 기준으로는 대형 teacher 2개를 활용해 student를 더 잘 가르치는 방향을 쓴다.
4. 최종 평가는 `Recall@K` 중심의 retrieval 성능으로 판단한다.
5. 학습 구조는 반복 실험이 쉬운 형태를 우선한다.

---

## 3. 현재 문맥의 운영 기본값

### 3.1 현재 기준 학생 모델

- 이미지 타워: `MobileNetV4 Hybrid Large`
- 텍스트 타워: `hf-hub:DatologyAI/retr-opt-vit-b-32`
- 구조: `DualTowerStudent`
- 이유: Apache-2.0, retrieval 최적화, OpenCLIP 호환, 대회 텍스트 입력 규격(`1x77`) 적합 조건을 함께 만족하는 현재 기본 후보이기 때문
- 현재 기본 구현 파일: `src/lpcvc_retrieval/dual_tower.py`
- 입력 스펙: 이미지 `384 x 384`, 텍스트 길이 `77`
- 최종 임베딩 차원: `256`
- 단, 학생 모델 역시 실험 결과와 대회 조건에 따라 교체될 수 있음

### 3.2 현재 기준 teacher 후보 조합

- `ViT-gopt-16-SigLIP2-256`
- `PE-Core-bigG-14-448`

선정 이유:
- 둘 다 retrieval 성능이 강한 상위권 공개 모델 계열
- `open_clip` 기반 로딩 경로와 잘 맞음
- offline feature extraction 구조에 태우기 쉬움
- 라이선스와 공개성 측면에서 현재 프로젝트 운영 조건과 충돌이 비교적 적음

주의:
- 이 조합은 `현재 기본 운영안`일 뿐이며, 더 나은 teacher 후보가 확인되면 바뀔 수 있음
- 따라서 이후 문서나 실험 로그를 볼 때는 "왜 이 조합을 잠정 채택했는가"로 이해하는 것이 맞음

### 3.3 현재 우선 두는 학습 방향

- `Offline Feature Distillation`을 현재 기본 운영안으로 둠
- `Temperature Scheduling`을 적용한 상태로 운영 중
- `Adaptive Teacher Weight`는 구현 완료 상태
- `EMA`는 유지하는 쪽에 무게를 둠
- `SWA`는 현재 제외한 상태
- `강한 Multi-crop`은 현재 제외한 상태

제외 이유:
- `SWA`는 EMA와 역할이 겹쳐 실험 해석이 흐려짐
- 강한 `Multi-crop`은 현재 offline teacher target과 의미 차이를 키울 수 있음

---

## 4. 프로젝트 전체 구조 지도

```text
Repository/
├─ README.md
├─ config.yaml
├─ run_train.py
├─ compile_and_profile.py
├─ requirements.txt
├─ pyproject.toml
├─ LICENSE
├─ THIRD_PARTY_LICENSES.md
├─ dataset/
├─ features/
├─ runs/
├─ scripts/
│  ├─ extract_features.py
│  └─ preprocess/
├─ src/
│  └─ lpcvc_retrieval/
└─ docs/
   ├─ README.md
   ├─ PROJECT_GUIDE.md
   ├─ PROJECT_MAP.md
   └─ archive/
```

### 4.1 루트 파일

- `README.md`
  - 실행 순서, 빠른 시작, 현재 기본 동작 설명
- `config.yaml`
  - 학습/증류/teacher/데이터/EMA 등 주요 설정
- `run_train.py`
  - 실제 학습 실행 진입점
- `compile_and_profile.py`
  - ONNX export 이후 Qualcomm AI Hub 쪽 경로 확인용
- `THIRD_PARTY_LICENSES.md`
  - 외부 모델/데이터/프레임워크 라이선스 정리

### 4.2 데이터/실험 산출물 폴더

- `dataset/`
  - 원본 또는 가공된 학습 데이터 위치
- `features/`
  - offline teacher feature `.pt` 저장 위치
- `runs/`
  - 학습 로그, 체크포인트, 실험 결과 저장 위치

### 4.3 스크립트 폴더

- `scripts/extract_features.py`
  - teacher 임베딩을 미리 추출하는 스크립트
- `scripts/preprocess/parse_lpcvc_sources.py`
  - COCO, Flickr30k, OpenImages, WIT 등을 공통 포맷으로 정리
- `scripts/preprocess/materialize_upload_subset.py`
  - 실제 업로드/학습에 필요한 subset 이미지 파일 구성

### 4.4 핵심 코드 폴더

- `src/lpcvc_retrieval/model.py`
  - 학생 모델 생성 및 로딩
- `src/lpcvc_retrieval/dual_tower.py`
  - 현재 학생 모델 구조 (`MobileNetV4 Hybrid Large` + `DatologyAI/retr-opt-vit-b-32`)
- `src/lpcvc_retrieval/data.py`
  - dataset, dataloader, transform 처리
- `src/lpcvc_retrieval/distill.py`
  - teacher distillation loss 계산
- `src/lpcvc_retrieval/train.py`
  - 전체 학습 루프
- `src/lpcvc_retrieval/losses.py`
  - contrastive 계열 손실 함수
- `src/lpcvc_retrieval/ema.py`
  - EMA 업데이트/적용/복원
- `src/lpcvc_retrieval/config.py`
  - 설정 로딩 및 config 객체 구조화

---

## 5. 실제 학습 흐름

### 5.1 큰 흐름

1. 데이터셋 정리
2. teacher feature 추출
3. student 학습
4. retrieval 평가
5. 필요 시 ONNX export / profile

### 5.2 조금 더 구체적으로

1. `parse_lpcvc_sources.py`로 여러 소스 데이터를 공통 형식으로 정리한다.
2. `extract_features.py`로 teacher 이미지/텍스트 임베딩을 미리 뽑는다.
3. `config.yaml`에 feature 경로와 teacher 설정을 맞춘다.
4. `run_train.py`로 student를 학습한다.
5. `runs/` 결과를 보고 실험 비교를 진행한다.

### 5.3 왜 offline feature 구조를 쓰는가

1. 대형 teacher를 매 step VRAM에 올리지 않기 위해서
2. batch를 더 키우기 위해서
3. 같은 teacher feature로 실험 반복을 쉽게 하기 위해서
4. dual teacher 실험에서 시간과 메모리를 절약하기 위해서

---

## 6. 현재 학습 구조의 뼈대

```text
원본 데이터
  -> dataset / jsonl 정리
  -> teacher feature 추출
  -> student batch 로딩
  -> main contrastive loss 계산
  -> distill loss 계산
  -> 합산 loss로 student 업데이트
  -> EMA shadow 업데이트
  -> validation / Recall@K 확인
```

### 6.1 teacher 관련 현재 포인트

1. teacher는 현재 2개 조합을 기본값으로 보고 있다.
2. offline 모드에서는 teacher를 매 step 직접 forward 하지 않는 구조를 선택할 수 있다.
3. adaptive teacher weighting을 통해 배치 상황에 따라 teacher 비중을 바꾸는 방향을 쓸 수 있다.
4. static weight는 adaptive를 끄는 경우에만 실제 mixing 기본값으로 의미가 커진다.

### 6.2 현재 성능 안정화 장치

1. `Temperature Scheduling`
2. `Adaptive Teacher Weight`
3. `Selective Distillation`
4. `EMA`
5. `Offline Feature Distillation`

---

## 7. 지금 이 프로젝트에서 특히 기억해야 할 것

### 7.1 가장 중요한 운영 원칙

1. 실험은 한 번에 하나의 변수만 바꾼다.
2. `Offline extraction`과 `training augment`는 분리해서 생각한다.
3. teacher feature를 다시 뽑아야 하는 조건을 함부로 섞지 않는다.
4. 결과 판단은 느낌이 아니라 `Recall@K`, 시간, VRAM, 변동폭으로 한다.

### 7.2 현재 기억해야 할 설정 포인트

1. teacher 조합은 현재 `SigLIP2 + PE-Core-bigG`를 기본값으로 보고 있음
2. adaptive teacher weight는 구현된 상태임
3. EMA는 현재 켜둔 쪽을 기본안으로 보고 있음
4. strong multi-crop은 현재 미적용 상태임
5. SWA는 현재 미적용 상태임

---

## 8. 최근 작업/대화 요약

이 섹션은 나중에 다시 봤을 때 "최근에 무엇을 고민했고, 무엇을 결정했는지"를 빠르게 복기하기 위한 세션 메모입니다.

### 8.1 최근 핵심 결정

1. 현재 저장소 구조는 유지하고 학생 모델만 현재 브랜치 기준으로 교체
2. offline feature extraction 구조를 핵심 운영 방식으로 채택
3. caption mismatch 문제를 구조적 결함으로 보고 수정 방향 검토 및 반영
4. `Temperature Scheduling`과 `Adaptive Teacher Weight`를 우선 적용 대상으로 정리
5. `EMA`는 유지, `SWA`는 보류
6. `Multi-crop 강화`는 현재 retrieval distillation 구조와 맞지 않아 보류
7. teacher 조합을 `SigLIP2 + MetaCLIP` 계열에서 다시 검토한 뒤 `SigLIP2 + PE-Core-bigG` 기준으로 정리
8. 문서 체계를 루트 + `docs/` + `archive/` 구조로 정리
9. 저장소 코드 라이선스는 Apache-2.0으로 정리하고, 외부 자산 라이선스는 별도 문서로 분리
10. student text encoder 기본값을 `DatologyAI/retr-opt-vit-b-32`로 교체하기로 결정
11. 작은 teacher(`ViT-B-32`, `ViT-B-16`) 기준 feature 추출, online/offline 학습, eval, ONNX export 재확인

### 8.2 최근 대화 주제 목록

1. offline feature extraction 구조의 필요성
2. teacher caption mismatch와 distillation noise 문제
3. temperature scheduling의 목적과 효과
4. adaptive teacher weight의 원리와 실제 의미
5. EMA와 SWA의 차이
6. multi-crop을 현재 프로젝트에 적용할지 여부
7. teacher 후보 비교
8. `SigLIP2 + PE-Core-bigG` 조합 선택
9. 문서 구조 정리와 handover 문서 보관 위치 정리
10. 라이선스 정리
11. `MobileNetV4 Hybrid Large + DatologyAI/retr-opt-vit-b-32` 학생 구조 smoke 검증 및 AI Hub upload-only 확인

### 8.3 지금 문맥에서 다시 시작할 때 먼저 떠올릴 것

1. 이 프로젝트는 현재 "모바일 경량 retrieval student를 dual teacher로 증류"하는 구조로 운영 중인 상태로 이해하면 된다.
2. 현재 기준 teacher 기본안은 `SigLIP2 + PE-Core-bigG`로 보고 있지만, 이것도 교체 가능한 조합이다.
3. 운영 방식의 중심은 현재 `offline feature + adaptive distill + EMA` 쪽에 있다.
4. 다음 실험 우선순위는 이미 구현된 기능을 안정적으로 검증하는 쪽에 가깝다.

---

## 9. 문서 읽는 순서

### 9.1 처음 복기할 때

1. `docs/PROJECT_MAP.md`
2. `README.md`
3. `docs/PROJECT_GUIDE.md`

### 9.2 최근 변경 이력을 볼 때

현재 브랜치에는 별도 handover/archive 문서를 커밋해 두지 않았습니다. 최근 변경 이력은 `git log`, `README.md`, `docs/PROJECT_GUIDE.md` 기준으로 확인합니다.

---

## 10. 간단 학습 로드맵

이 섹션은 "지금 다시 시작하면 무엇부터 해야 하는가"를 아주 짧게 정리한 실전 순서입니다.

### 10.1 기본 로드맵

1. 데이터셋/jsonl 상태 확인
2. teacher 후보와 현재 기본 조합 확인
3. offline feature 추출 여부 확인
4. baseline 학습 1회 실행
5. temperature scheduling 포함 버전 확인
6. adaptive teacher weight on/off 비교
7. 필요 시 다음 실험으로 이동

### 10.2 현재 프로젝트 기준 추천 순서

1. `Baseline`
   - 현재 설정이 정상 동작하는지 먼저 확인
2. `Offline Feature`
   - teacher feature를 현재 조합 기준으로 준비
3. `Temperature Scheduling`
   - 초반 학습 안정화 확인
4. `Adaptive Teacher Weight`
   - 고정 비율 대비 이득 확인
5. `False-Negative 보정`
   - 그 다음 단계에서 검토

### 10.3 지금 당장 기억할 실행 우선순위

1. 먼저 baseline이 정상인지 본다.
2. 그다음 offline feature가 현재 teacher 기준으로 맞는지 본다.
3. 그다음 temperature scheduling이 포함된 학습을 본다.
4. adaptive teacher weight는 baseline이 잡힌 뒤 비교한다.
5. 고난도 기법은 마지막에 본다.

---

## 11. 다음에 접속했을 때 바로 확인할 체크리스트

- [ ] 지금 teacher 조합이 여전히 `SigLIP2 + PE-Core-bigG`인지 확인
- [ ] `config.yaml`의 distill 관련 기본값이 내가 의도한 상태인지 확인
- [ ] `features/`에 현재 teacher 기준 offline feature가 준비되어 있는지 확인
- [ ] `runs/` 안에서 가장 최근 실험이 무엇이었는지 확인
- [ ] 다음 실험이 baseline 확인인지, adaptive 검증인지, false-negative 보정인지 확인

---

## 12. 이 문서의 역할 요약

이 문서는 상세 설계서가 아닙니다.  
이 문서는 "현재 프로젝트의 방향, 구조, 최근 결정, 다음 출발점"을 빠르게 복원하기 위한 지도입니다.

즉:

1. 길을 잃었을 때 먼저 보는 문서
2. 내가 무엇을 하던 중이었는지 떠올리는 문서
3. 세부 구현 문서로 내려가기 전의 상위 개요 문서


