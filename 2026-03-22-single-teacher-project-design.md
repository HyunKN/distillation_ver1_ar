# Single Teacher Project Design for Beginners

작성일: 2026-03-22  
대상 프로젝트: `distillation_ver1`  
현재 결정: `ViT-gopt-16-SigLIP2-256` single teacher를 기본 전략으로 사용

---

## 0. 이 문서는 누구를 위한 문서인가

이 문서는 `이 분야를 아직 잘 모르는 사람`, 즉 딥러닝과 데이터과학을 막 공부하기 시작한 사람도 이해할 수 있도록 쓴 프로젝트 안내서입니다.

그래서 이 문서는 일부러 아래 순서로 설명합니다.

1. 먼저 어려운 용어를 쉽게 설명합니다.
2. 그다음 우리 프로젝트가 정확히 무엇을 하는지 설명합니다.
3. 왜 single teacher 방식으로 방향을 바꿨는지 설명합니다.
4. 실제로 코드를 왜 그렇게 수정했는지 설명합니다.
5. 논문과 피드백에서 무엇을 가져오고 무엇을 버릴지 설명합니다.
6. 프로젝트를 체계적으로 하려면 무엇을 기록하고 무엇을 평가해야 하는지 설명합니다.
7. `autoresearch` 같은 자동 연구 방식이 우리에게 도움이 되는지 설명합니다.

이 문서의 목표는 단순합니다.

- 코드를 바꾸는 것 자체보다
- `왜 그렇게 하는지 이해하고`
- `나중에 결과를 설명할 수 있게 만드는 것`

입니다.

---

## 1. 이 프로젝트를 아주 쉽게 설명하면

이 프로젝트는 `사진`과 `문장`을 서로 잘 찾게 만드는 모델을 만드는 프로젝트입니다.

예를 들어:

- 사진을 넣으면 그 사진을 잘 설명하는 문장을 찾아야 합니다.
- 문장을 넣으면 그 문장에 맞는 사진을 찾아야 합니다.

이걸 `image-text retrieval`이라고 합니다.

쉽게 말하면:

- 사진과 문장을 같은 의미 공간에 잘 배치해서
- 서로 가까운 것끼리 잘 만나게 만드는 문제

입니다.

### 한 줄 비유

사진과 문장이 각각 다른 언어를 쓰고 있다고 생각하면 됩니다.

- 사진은 "이미지 언어"
- 문장은 "텍스트 언어"

를 씁니다.

우리 모델의 역할은 이 둘을 같은 뜻끼리 가까이 놓는 `통역사` 같은 역할입니다.

---

## 2. 먼저 알아야 하는 아주 기초적인 용어

이 분야를 처음 보면 용어가 가장 어렵습니다.  
그래서 우리 프로젝트에서 꼭 나오는 말만 먼저 쉽게 설명합니다.

### 2.1 student model

`student model`은 우리가 실제로 키우고 싶은 모델입니다.

비유하면:

- teacher는 가르치는 선생님
- student는 배우는 학생

입니다.

우리 프로젝트에서 student는 실제로 나중에 배포하고 싶은 가벼운 모델입니다.

### 2.2 teacher model

`teacher model`은 student보다 보통 더 크고, 더 강한 모델입니다.

teacher는 정답을 직접 주는 것보다,

- 어떤 사진이 어떤 문장과 더 잘 맞는지
- 어떤 방향으로 student가 학습하면 좋을지

를 알려주는 역할을 합니다.

### 2.3 distillation

`distillation`은 teacher가 student를 가르치는 방식입니다.

쉽게 말하면:

- teacher가 이미 잘 알고 있는 패턴을
- student가 더 작고 빠른 구조 안에서 배우게 만드는 과정

입니다.

즉, distillation은 `작은 모델이 큰 모델에게 배우는 학습 방식`입니다.

### 2.4 retrieval

`retrieval`은 "찾아오기"입니다.

우리 프로젝트에서는 두 가지가 중요합니다.

- I2T: Image-to-Text  
  사진을 보고 맞는 문장을 찾는 것
- T2I: Text-to-Image  
  문장을 보고 맞는 사진을 찾는 것

### 2.5 baseline

`baseline`은 비교의 기준점입니다.

예를 들어:

- teacher 없이 학습한 모델
- single teacher로 가장 단순하게 학습한 모델

이런 것이 baseline이 됩니다.

baseline이 없으면,

- 좋아졌는지
- 나빠졌는지
- 그냥 우연인지

를 판단할 수 없습니다.

즉, baseline은 `출발점`입니다.

### 2.6 smoke test

`smoke test`는 "성능이 좋나?"를 보는 테스트가 아닙니다.

이건 그냥:

- 코드가 돌아가나
- 에러 없이 끝나나
- 학습/평가가 정상 작동하나

를 보는 가장 기초적인 확인입니다.

즉, smoke test는 `건강검진` 같은 것입니다.

### 2.7 ablation

`ablation`은 어떤 요소 하나를 켜고 끄면서 그 요소가 정말 도움이 되는지 보는 실험입니다.

예를 들어:

- hard negative loss를 끄고 실험
- hard negative loss를 켜고 실험

이렇게 비교하면 "이 요소가 정말 효과가 있는가?"를 볼 수 있습니다.

즉, ablation은 `한 요소씩 분리해서 검증하는 실험`입니다.

### 2.8 metric

`metric`은 성능을 숫자로 나타내는 기준입니다.

우리 프로젝트에서는 주로 아래를 봅니다.

- I2T Recall@1, Recall@5, Recall@10
- T2I Recall@1, Recall@5, Recall@10
- train loss
- distill loss

여기서 Recall@10은 쉽게 말하면:

- 정답이 상위 10개 안에 들어왔는가

를 보는 지표입니다.

### 2.9 train / validation / test

이것도 아주 중요합니다.

- train set: 모델을 학습시키는 데이터
- validation set: 실험 중간 비교용 데이터
- test set: 마지막 최종 확인용 데이터

처음 하는 사람이 자주 하는 실수는 `test를 자꾸 들여다보며 튜닝하는 것`입니다.

그러면 test도 사실상 연습장이 되어버립니다.

그래서 보통은:

- train으로 학습하고
- validation으로 비교하고
- test는 마지막에만 씁니다.

### 2.10 overfitting

`overfitting`은 train data에는 너무 잘 맞는데, 새로운 데이터에는 잘 안 맞는 상태입니다.

예를 들면:

- train loss는 계속 좋아짐
- validation 성능은 오히려 안 좋아짐

이러면 overfitting 가능성을 의심합니다.

### 2.11 confounder

이건 입문자에게 가장 어려운 말 중 하나인데, 뜻은 단순합니다.

`실험 결과를 헷갈리게 만드는 숨은 변수`입니다.

예를 들어 한 번에 아래를 다 바꾸면:

- teacher도 바꾸고
- loss도 바꾸고
- dataset도 바꾸고
- batch size도 바꾸고

성능이 달라져도 무엇 때문인지 모르게 됩니다.

이런 혼란을 만드는 것이 confounder입니다.

쉽게 말해:

- `한 번에 너무 많이 바꾸면 원인을 모른다`

는 뜻입니다.

---

## 3. 우리 프로젝트는 정확히 어떤 구조인가

현재 우리 프로젝트는 아래 구조입니다.

- student: `MobileNetV4 Hybrid Large` 이미지 타워 + `DatologyAI retr-opt-vit-b-32` 텍스트 타워
- teacher: `ViT-gopt-16-SigLIP2-256`
- 문제: image-text retrieval
- 목표: 성능을 높이면서도 모바일 배포 가능성을 유지

즉, 우리는 거대한 모델을 만드는 것이 목적이 아닙니다.

우리가 원하는 것은:

- 작은 학생 모델이
- 충분히 좋은 성능을 내고
- 나중에 ONNX export와 모바일 경로도 유지되는 것

입니다.

### 아주 쉽게 말한 현재 전략

현재 전략은 아래 한 줄로 정리할 수 있습니다.

`복잡한 dual teacher 실험보다, single teacher로 이해 가능한 기준점을 먼저 만든다.`

---

## 4. 왜 single teacher로 방향을 바꿨는가

이 부분이 현재 프로젝트에서 가장 중요합니다.

### 4.1 예전 기본 구조는 왜 어려웠는가

예전 기본 설정은 대략 이런 느낌이었습니다.

- teacher가 두 개
- teacher를 상황에 따라 섞음
- adaptive teacher weighting 사용
- 추가 loss 여러 개 사용

겉으로 보면 뭔가 강력해 보입니다.  
하지만 실제 프로젝트 운영에서는 문제가 많습니다.

#### 문제 1. 너무 많은 것이 동시에 바뀐다

성능이 좋아져도:

- teacher가 둘이라서 좋아진 건지
- adaptive weighting 때문인지
- extra loss 때문인지

구분하기 어렵습니다.

#### 문제 2. 설명하기 어렵다

팀 프로젝트에서는 "결과가 좋다"만으로는 부족합니다.

다른 사람에게 아래를 설명할 수 있어야 합니다.

- 왜 이렇게 설계했는가
- 왜 이 결과가 나왔는가
- 무엇이 진짜 효과였는가

dual teacher adaptive는 이 설명이 어렵습니다.

#### 문제 3. 일정이 촉박하다

지금은 대학원 논문처럼 아주 긴 시간 동안 깊게 파고드는 상황이 아닙니다.

그래서 지금은:

- 제일 복잡한 전략

보다

- 제일 해석 가능한 전략

이 더 중요합니다.

### 4.2 single teacher가 좋은 이유

single teacher는 아래 장점이 있습니다.

- 구조가 단순합니다.
- 실험 해석이 쉽습니다.
- 실패 원인 찾기가 쉽습니다.
- baseline 만들기가 쉽습니다.
- 팀 전체가 이해하기 쉽습니다.

즉, single teacher는 "덜 멋져 보여도" 프로젝트를 앞으로 가게 만드는 전략입니다.

### 4.3 왜 SigLIP2 teacher를 선택했는가

지금 task는 image-text retrieval입니다.

즉, 중요한 것은:

- 이미지와 문장의 의미를 잘 맞추는 능력

입니다.

피드백과 정리 문서를 보면, 현재 상황에서는 SigLIP2가 이 목적에 가장 잘 맞는 teacher로 판단됩니다.

쉽게 말하면:

- 지금은 "시각적 강건성"보다
- "이미지-문장 정렬 품질"

이 더 중요하다고 본 것입니다.

---

## 5. 코드를 왜 그렇게 수정했는가

이제 실제 수정 내용을 초심자 기준으로 설명합니다.

### 5.1 `config.yaml`을 바꾼 이유

이 파일은 프로젝트의 기본 실행 설정입니다.

즉, `python run_train.py --config config.yaml`을 실행했을 때 가장 먼저 따라가는 길입니다.

그래서 기본 설정은 프로젝트의 "기본 철학"을 보여줍니다.

#### 바꾼 핵심 1. teacher를 한 개만 남겼다

이유:

- baseline을 해석 가능하게 만들기 위해
- distillation 경로를 단순하게 만들기 위해
- dual teacher의 복잡성을 일단 걷어내기 위해

#### 바꾼 핵심 2. `adaptive_teacher_weight`를 껐다

이유:

- teacher가 하나면 adaptive weighting이 거의 의미가 없습니다.
- baseline은 "교사 선택 실험"이 아니라 "교사 하나로 잘 배우는가"를 보는 단계입니다.

#### 바꾼 핵심 3. `teacher_weight_mode`를 `static`으로 했다

이유:

- baseline에서는 teacher를 고정하는 것이 가장 단순합니다.
- 흔들리는 선택 규칙보다 고정된 기준이 비교에 유리합니다.

#### 바꾼 핵심 4. `affinity_temp_schedule`을 `constant`로 했다

이유:

- baseline에서는 temperature까지 실험 변수로 만들고 싶지 않았습니다.
- 단순한 기준점을 먼저 만든 뒤, 나중에 필요하면 다시 만지는 편이 좋습니다.

#### 바꾼 핵심 5. `w_rank`, `w_hard_negative`, `w_text_text`를 0으로 했다

이유:

- 좋은 아이디어라도 한 번에 많이 넣으면 원인 분리가 어렵습니다.
- baseline에서는 "teacher가 student를 잘 가르치고 있는가"를 먼저 봐야 합니다.
- extra loss는 baseline이 잡힌 뒤에 하나씩 다시 켜는 것이 더 낫습니다.

#### 바꾼 핵심 6. `allowed_sources`를 `coco + flickr30k`로 넣었다

이유:

- 가장 안정적인 데이터 조합으로 baseline을 만들기 위해
- noisy source를 초기에 빼서 해석을 쉽게 하기 위해

#### 바꾼 핵심 7. `device: auto`로 바꿨다

이유:

- 특정 환경에만 묶이지 않게 하려고
- 실행 환경에 따라 CPU/GPU를 더 유연하게 쓰게 하려고

#### 바꾼 핵심 8. `use_compile: false`로 바꿨다

이유:

- 초기에 compile 문제와 모델 문제를 섞고 싶지 않았기 때문입니다.
- baseline 단계에서는 속도보다 안정적인 관찰이 더 중요합니다.

### 5.2 smoke config를 따로 만든 이유

아래 파일들을 따로 만든 이유는 "문제 원인을 잘 나누기 위해서"입니다.

- `configs/no_teacher_smoke.yaml`
- `configs/single_teacher_smoke.yaml`
- `configs/single_teacher_baseline.yaml`

#### `no_teacher_smoke.yaml`

이건 teacher 없이 student만 돌려보는 설정입니다.

이걸 먼저 하는 이유는:

- 데이터 로딩이 되는지
- 기본 학습 루프가 도는지
- eval이 되는지

를 teacher와 상관없이 확인하기 위해서입니다.

즉:

- 학생 혼자 걷는지 먼저 본다

는 뜻입니다.

#### `single_teacher_smoke.yaml`

이건 teacher를 붙인 뒤 distillation 경로가 정상 작동하는지 보는 설정입니다.

즉:

- 학생 혼자 걷는 건 되는데
- 선생님 붙였을 때만 문제가 생기는지

를 분리해서 보기 위한 것입니다.

#### `single_teacher_baseline.yaml`

이건 smoke가 아니라 실제 baseline 실험용입니다.

즉:

- smoke: 돌아가는지 확인
- baseline: 성능의 기준점 확보

입니다.

### 5.3 `data.py`에 `allowed_sources`를 넣은 이유

이건 이번 수정 중에서 가장 실험 설계와 직접 연결되는 부분입니다.

기존에는 JSONL을 읽을 때 source를 선택적으로 거르는 기능이 없었습니다.

그러면 아래 같은 실험을 쉽게 못 합니다.

- `coco + flickr30k`만 쓰기
- `open_images` 추가하기
- `wit` 제외하기

즉, "데이터 조합"을 실험 변수로 다루기가 어려웠습니다.

그래서 `allowed_sources`를 추가했습니다.

이 기능의 의미는 단순합니다.

- 이제 데이터 source를 실험 변수로 쓸 수 있다

이건 나중에 다음 실험으로 이어집니다.

- baseline: `coco + flickr30k`
- 확장 1: `+ open_images`
- 확장 2: `+ length filtering`

### 5.4 `train.py` 로깅을 강화한 이유

입문자일수록 "모델이 돌았다"와 "무슨 일이 있었는지 안다"를 헷갈리기 쉽습니다.

하지만 프로젝트에서는 둘이 다릅니다.

모델이 돌아가도 아래를 모르면 실험이 아닙니다.

- loss가 줄었는가
- teacher 신호가 들어왔는가
- validation metric이 좋아졌는가

그래서 다음 로깅을 보강했습니다.

- `train/distill_loss`
- `train/logit_scale`
- `train/distill_temp`
- `val` retrieval metric

즉, 목적은 아래입니다.

- 나중에 "왜 이런 결과가 나왔지?"를 설명할 수 있게 하기

### 5.5 `run_train.py`와 `README.md`를 손본 이유

코드만 바꾸고 설명을 안 바꾸면 팀원은 예전 전략으로 이해할 수 있습니다.

특히 아래 같은 혼란이 생깁니다.

- 지금 기본 전략이 dual인가 single인가
- 어떤 config를 먼저 실행해야 하는가
- 실험 순서가 무엇인가

그래서 문서와 실행 안내도 같이 맞췄습니다.

이건 단순 꾸미기가 아니라:

- 팀이 같은 방향을 보게 만드는 작업

입니다.

---

## 6. 논문과 피드백 문서에서 무엇을 가져가고 무엇을 버릴 것인가

여기서 중요한 건 "좋은 아이디어"와 "지금 당장 우리 프로젝트에 맞는 아이디어"를 구분하는 것입니다.

### 6.1 지금 바로 가져갈 것

#### 1. single teacher 기본 전략

이건 이미 채택했습니다.

이유:

- 단순함
- 해석 가능성
- 일정 현실성

#### 2. `coco + flickr30k` baseline

이것도 바로 가져갈 전략입니다.

이유:

- 현재 retrieval 평가 구조와 잘 맞음
- caption 스타일이 상대적으로 안정적임
- baseline 결과 해석이 쉬움

#### 3. `open_images`는 나중에 filtered 데이터로 비교

이것도 가져갈 생각입니다.

하지만 지금은 baseline에 바로 넣지 않습니다.

이유:

- 장문 caption이 많음
- 배경 설명이 과함
- 해석을 흐릴 수 있음

즉:

- `기본 출발점`이 아니라
- `추가 비교용 데이터`

로 보는 것이 맞습니다.

#### 4. `wit`는 초기 baseline에서 제외

이것도 가져갈 판단입니다.

이유:

- 다국어
- 제목형 문장
- 현재 retrieval task와 결이 다를 수 있음

즉:

- 지금 단계에서는 이득보다 혼란이 클 수 있음

으로 봅니다.

#### 5. MobileCLIP / MobileCLIP2에서 recipe를 배우는 것

이건 가져갈 가치가 큽니다.

다만 오해하면 안 되는 점이 있습니다.

지금 바로:

- 학생 모델을 통째로 갈아엎자

가 아니라,

- 어떤 teacher를 쓸지
- 어떤 데이터를 쓸지
- 학습을 어떤 순서로 할지

같은 `학습 운영 방식`을 배우자는 뜻입니다.

#### 6. JEST / SIEVE식 데이터 선택 생각

이것도 좋은 방향입니다.

쉽게 말하면:

- 모든 데이터를 똑같이 쓰지 말고
- 더 도움이 되는 데이터를 더 우선해서 쓰는 생각

입니다.

현재 우리 파이프라인에는 이미:

- source 분리
- source cap
- dedupe
- split

이 있으므로, 다음 단계로 연결하기 좋습니다.

#### 7. 실험을 구조적으로 기록하자는 생각

이건 논문 아이디어보다 더 중요할 수 있습니다.

왜냐하면 프로젝트는 결국 아래가 남아야 하기 때문입니다.

- 무엇을 바꿨는가
- 왜 바꿨는가
- 결과가 어땠는가
- keep 할 것인가, 버릴 것인가

### 6.2 나중에 다시 볼 것

#### 1. hard negative

이미 코드에는 구현돼 있습니다.

하지만 지금 당장 켜지 않는 이유는:

- baseline 해석이 흐려지기 때문입니다.

즉:

- 나쁜 아이디어라서가 아니라
- `지금 순서가 아니다`

라는 뜻입니다.

#### 2. text-text loss

이것도 이미 들어 있습니다.

하지만 baseline 이후에 다시 보는 것이 좋습니다.

이유는 hard negative와 같습니다.

#### 3. offline feature mode

이것도 좋은 기능입니다.

장점:

- teacher VRAM 절약
- 반복 실험 속도 향상

하지만 지금은 baseline이 먼저입니다.

baseline이 없으면 offline의 이득도 제대로 설명하기 어렵습니다.

#### 4. TULIP 확장

현재는 text-text 정도가 일부 들어 있는 상태입니다.

더 복잡한 consistency loss는:

- baseline이 자리 잡고
- extra loss 실험이 안정화된 뒤

보는 것이 좋습니다.

### 6.3 지금은 버릴 것

#### 1. dual teacher adaptive를 주력 전략으로 삼는 것

이건 지금은 버리는 게 맞습니다.

이유:

- 복잡함
- 해석 어려움
- 일정 부담
- teacher 분포 차이를 잘못 읽을 위험

#### 2. baseline 없이 고급 튜닝부터 시작하는 것

예:

- mixing 비율 최적화
- temp 최적화
- loss 여러 개 동시에 추가

이건 초심자가 가장 빠르게 길을 잃는 패턴입니다.

#### 3. 문제 정의를 바꾸는 것

예:

- 객체 단위 평가
- region-level retrieval
- box 기반 실험

현재 프로젝트는 image-caption retrieval입니다.

즉:

- 현재 문제를 잘 푸는 것

이 먼저이고,

- 다른 문제를 새로 만드는 것

은 지금 할 일이 아닙니다.

---

## 7. 프로젝트를 체계적으로 하려면 무엇을 알아야 하는가

이 파트는 "연구를 어떻게 하는가"를 아주 기초적으로 설명하는 부분입니다.

### 7.1 연구를 잘한다는 것은 무엇인가

처음에는 연구나 프로젝트를 잘한다는 것이

- 논문을 많이 읽는 것
- 어려운 모델을 쓰는 것

이라고 느껴질 수 있습니다.

하지만 실제로는 아래에 더 가깝습니다.

- 문제를 분명히 정한다
- 기준점을 만든다
- 한 번에 하나씩 바꾼다
- 결과를 기록한다
- 결과를 설명할 수 있다

즉, 좋은 연구는 `똑똑한 추측`보다 `좋은 비교와 기록`에서 나옵니다.

### 7.2 실제 프로젝트는 보통 어떤 순서로 가는가

#### 1. 문제 정의

먼저 아래를 적어야 합니다.

- 우리는 무엇을 만들고 싶은가
- 무엇이 중요하고 무엇이 덜 중요한가
- 어떤 제약이 있는가

우리 프로젝트라면:

- 목표: 모바일에서도 쓸 수 있는 retrieval 모델
- 중요: retrieval accuracy
- 제약: export와 모바일 경로 유지

#### 2. baseline 만들기

그다음 가장 단순한 기준점을 만듭니다.

이게 있어야 이후 실험을 비교할 수 있습니다.

#### 3. 가설 세우기

예:

- single teacher가 no-teacher보다 낫다
- open_images를 추가하면 성능이 오른다
- hard negative가 도움이 된다

이런 식으로 한 번에 하나의 질문만 다룹니다.

#### 4. 실험하기

가설에 맞는 실험을 돌립니다.

#### 5. 결과 기록하기

이 단계가 정말 중요합니다.

기록이 없으면 실험이 아니라 그냥 "어렴풋한 기억"이 됩니다.

#### 6. keep/discard 판단하기

결과를 보고 아래를 정합니다.

- 이 설정을 유지할까
- 버릴까
- 다음엔 무엇을 볼까

### 7.3 우리 프로젝트에서 꼭 체크해야 하는 것

이건 실제 체크리스트처럼 생각하면 됩니다.

#### A. 시스템 체크

- config가 정상 로드되는가
- 데이터 경로가 맞는가
- train loop가 도는가
- eval이 끝까지 되는가
- checkpoint가 저장되는가
- teacher 로딩이 되는가
- distill loss가 계산되는가
- NaN이 없는가

#### B. 데이터 체크

- caption이 이미지 전체를 설명하는가
- retrieval에 적합한 문장인가
- 너무 긴가
- source마다 스타일 차이가 너무 큰가
- validation split이 너무 치우치지 않았는가

#### C. 학습 체크

- train loss가 줄어드는가
- distill loss가 줄어드는가
- validation metric이 같이 좋아지는가
- 학습이 불안정하지 않은가

#### D. 결과 체크

- baseline보다 좋아졌는가
- 왜 좋아졌는지 설명 가능한가
- complexity가 늘어난 만큼 가치가 있는가

#### E. 배포 체크

- ONNX export가 되는가
- split encoder export가 되는가
- compile/profile 경로가 유지되는가

---

## 8. 무엇을 기록해야 프로젝트가 체계적이 되는가

초심자에게 가장 실용적인 부분입니다.

실험은 기억에 의존하면 안 됩니다.  
적어도 아래는 남겨야 합니다.

### 8.1 실험 전에 적을 것

- 실험 이름
- 날짜
- 가설
- 바꾼 것
- 안 바꾼 것
- 사용할 config

예:

- 실험명: `single_teacher_smoke_01`
- 가설: single teacher 경로가 정상 작동한다
- 바꾼 것: teacher on
- 안 바꾼 것: dataset은 coco만
- config: `configs/single_teacher_smoke.yaml`

### 8.2 실험 중에 볼 것

- 에러가 없는가
- loss가 NaN이 아닌가
- teacher가 정상 로드됐는가
- distill loss가 찍히는가
- eval이 도는가

### 8.3 실험 후에 적을 것

- 최종 metric
- train loss
- distill loss
- 문제점
- keep/discard
- 다음 실험 아이디어

예:

- 결과: eval 성공, distill loss 계산 정상
- 판단: keep
- 다음: baseline으로 확장

### 8.4 추천 문서 구조

우리 프로젝트에서는 최소한 아래 정도가 있으면 좋습니다.

- `research/PROJECT_PLAN.md`
- `research/EXPERIMENTS.tsv`
- `research/DECISIONS.md`

쉽게 말하면:

- 계획서
- 실험표
- 왜 그렇게 결정했는지 기록

이 세 가지입니다.

---

## 9. `autoresearch`는 무엇이고, 우리에게 도움이 되는가

이제 외부 기술을 아주 쉽게 설명합니다.

### 9.1 `karpathy/autoresearch`를 쉽게 설명하면

이 아이디어는 아주 단순합니다.

- 사람이 매번 손으로 하나하나 실험하지 않고
- 에이전트가 코드를 조금 바꾸고
- 실험을 돌리고
- 점수가 좋아졌는지 보고
- 좋으면 유지하고, 아니면 버리는 방식

입니다.

쉽게 말하면:

- `자동 실험 반복 시스템`

입니다.

### 9.2 이 방식이 좋은 이유

좋은 점은 아래입니다.

- baseline을 먼저 잡게 만듭니다.
- 결과 비교를 강제합니다.
- keep/discard를 분명히 하게 만듭니다.
- 실험 기록이 잘 남습니다.

즉, 좋은 실험 습관을 몸으로 익히게 해 줍니다.

### 9.3 `autoresearch-skill`은 무엇인가

이건 비슷한 아이디어를 `스킬 프롬프트 개선`에 적용한 것입니다.

즉:

- 어떤 스킬이 있고
- 그 스킬이 잘 작동하는지 테스트하고
- 조금씩 바꾸고
- 점수가 좋아지면 유지하는 방식

입니다.

여기서 배울 만한 핵심은 아래입니다.

- baseline 먼저
- binary yes/no 체크
- mutation 기록
- results 표

### 9.4 우리 프로젝트에 그대로 가져와도 되는가

현재는 `그대로는 추천하지 않습니다`.

이유는 아래와 같습니다.

#### 이유 1. 우리 프로젝트는 더 복잡하다

Karpathy 방식은 작은 실험 repo에 특히 잘 맞습니다.

우리 프로젝트는:

- config
- data
- distill
- train
- export

가 분리되어 있고,

실험도 더 복합적입니다.

#### 이유 2. metric이 하나로 딱 고정되지 않는다

우리는 아래를 같이 봐야 합니다.

- I2T
- T2I
- distill loss
- export 가능 여부
- 모바일 제약

즉, "숫자 하나로 자동 판단"하기가 아직 어렵습니다.

#### 이유 3. 지금은 사람의 이해가 더 중요하다

현재 단계에서는 자동 루프보다 아래가 더 중요합니다.

- 팀이 이해하는가
- baseline이 안정적인가
- 데이터 선택이 타당한가

즉:

- 자동 연구 시스템을 도입하기 전에
- 사람이 프로젝트를 이해하고 정리하는 단계

가 먼저입니다.

### 9.5 그래도 우리가 배워야 할 것은 많다

그대로 쓰진 않더라도, 아래는 적극적으로 가져오면 좋습니다.

- baseline-first
- one-change-at-a-time
- keep/discard
- 실험 로그
- binary checklist

즉, 결론은 이렇습니다.

- `full autoresearch`: 지금은 이르다
- `autoresearch식 운영 원칙`: 매우 유용하다

이걸 저는 `autoresearch-lite`라고 부를 수 있다고 봅니다.

즉:

- 자동 self-modifying loop는 아직 하지 않지만
- 실험 운영 방식은 autoresearch처럼 더 체계적으로 하자

는 뜻입니다.

---

## 10. 지금 우리 프로젝트의 최종 방향

현재 기준으로 가장 좋은 방향은 아래입니다.

### 반드시 유지할 것

- SigLIP single teacher 기본 전략
- `coco + flickr30k` baseline
- no-teacher -> single-teacher -> baseline 순서
- 기록 가능한 실험 운영
- 한 번에 한 가지 큰 변화만 보는 원칙

### 나중에 확장할 것

- `open_images` filtered 추가
- hard negative ablation
- text-text ablation
- offline feature mode
- JEST/SIEVE식 데이터 subset 선택

### 지금은 하지 않을 것

- dual teacher adaptive 주력화
- `wit`를 baseline에 바로 넣기
- 객체 단위 문제로 바꾸기
- 여러 실험 변수를 한 번에 섞기
- full autoresearch self-modifying loop 도입

---

## 11. 바로 다음에 무엇을 하면 되는가

초심자 기준으로 아주 단순하게 다음 순서를 추천합니다.

1. `no_teacher_smoke` 실행  
   목적: teacher 없이 기본 학습 경로가 살아 있는지 확인

2. `single_teacher_smoke` 실행  
   목적: teacher를 붙였을 때 distillation 경로가 살아 있는지 확인

3. `single_teacher_baseline` 실행  
   목적: 비교 기준점 만들기

4. 실험 기록 파일 만들기  
   목적: 나중에 결과를 설명할 수 있게 하기

5. baseline 결과 정리하기  
   목적: 이후 모든 실험이 비교할 기준 확보

6. 그다음 `open_images` filtered 실험 설계  
   목적: 데이터 확장 효과 확인

7. 그다음 hard negative / text-text ablation  
   목적: 보조 loss 효과 분리 검증

---

## 12. 이 문서의 가장 중요한 한 줄

지금 이 프로젝트에서 가장 중요한 것은

`복잡한 아이디어를 많이 넣는 것`이 아니라,
`single teacher baseline을 먼저 만들고, 무엇을 왜 바꿨는지 기록하면서 한 단계씩 비교하는 것`

입니다.
