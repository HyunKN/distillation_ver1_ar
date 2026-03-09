# Student Text Encoder Migration Patch Notes

작성일: 2026-03-10

## 1. 이 패치가 필요한 이유

이번 패치의 출발점은 단순합니다.  
student의 text encoder를 기존 EVA 계열에서 `DatologyAI/retr-opt-vit-b-32`로 바꾸기로 결정했는데, 저장소 전체는 아직 그 결정이 완전히 반영된 상태가 아니었습니다.

겉으로는 `config.yaml`의 모델 이름만 바꾸면 끝나 보일 수 있습니다. 하지만 실제로는 그렇지 않습니다.

- 학습 데이터가 텍스트를 몇 토큰 길이로 자르는지
- ONNX export가 어떤 입력 shape로 나가는지
- 평가 스크립트가 현재 repo 코드를 읽는지 다른 설치본을 읽는지
- 문서가 지금 코드와 같은 설명을 하고 있는지
- smoke 테스트가 실제로 끝까지 도는지

이런 것들이 모두 함께 맞아야 "text encoder를 교체했다"라고 말할 수 있습니다.

즉 이번 패치는 모델 이름만 바꾼 작업이 아니라, 아래를 한 번에 정리한 작업입니다.

1. student text encoder 기본값 교체
2. 교체에 따라 달라지는 코드 경로 정리
3. 문서 최신화
4. 실제 동작 검증

## 2. 한눈에 보는 변경 전 / 변경 후

### 변경 전

- 문서상으로는 DatologyAI text tower를 쓰는 방향이 정리돼 있었음
- 하지만 실제 기본 설정은 EVA 계열 기준 흔적이 남아 있었음
- 텍스트 길이 처리가 일부 경로에서 사실상 `77` 하드코딩에 가까웠음
- `eval.py`, `export_onnx_split.py`는 현재 repo가 아니라 다른 설치된 패키지를 읽을 위험이 있었음
- EMA로 best를 고를 때, "평가한 가중치"와 "저장된 best.pt"가 다를 수 있었음
- `train_augment=false`일 때 train 이미지 크기가 설정값이 아닌 224로 내려가는 버그가 있었음

### 변경 후

- 기본 student text encoder가 `hf-hub:DatologyAI/retr-opt-vit-b-32`로 통일됨
- dataset tokenization과 ONNX export가 모델의 실제 text context length를 따름
- eval/export 스크립트가 현재 repo의 코드를 우선해서 읽음
- EMA 기준 best 평가 결과와 저장 checkpoint 의미가 일치함
- train/eval/offline/export/upload 흐름을 smoke 테스트로 끝까지 확인함
- README와 GUIDE가 현재 코드 동작을 기준으로 설명하도록 정리됨

쉽게 말하면, 이번 패치로 저장소가 "문서만 맞는 상태"가 아니라 "설정, 코드, 문서, 테스트 결과가 같은 방향을 바라보는 상태"가 됐습니다.

## 3. 핵심 목표

이번 패치의 핵심 목표는 두 가지였습니다.

### 목표 1. student text encoder를 실제 기본값으로 교체

이제 이 저장소에서 별도 override 없이 기본 학습을 돌리면 student text encoder는 `DatologyAI/retr-opt-vit-b-32`를 사용합니다.

### 목표 2. 교체 후 전체 흐름이 정말로 동작하는지 확인

이번에는 코드만 바꾸고 끝내지 않았습니다. 아래 흐름을 직접 확인했습니다.

1. online distillation train
2. offline teacher feature extraction
3. offline distillation train
4. eval
5. ONNX split export
6. Qualcomm AI Hub upload-only

즉 "이론상 될 것 같다"가 아니라, 최소한의 smoke 기준으로는 실제로 끝까지 이어지는지 확인했습니다.

## 4. 코드 변경 상세

아래는 파일별로 `무엇이 바뀌었는지`와 `왜 바꿨는지`를 나눠 설명한 내용입니다.

### 4.1 `config.yaml`

#### 무엇이 바뀌었는가

- `model.text_model_name`을 `hf-hub:DatologyAI/retr-opt-vit-b-32`로 변경했습니다.
- `text_pretrained` 항목은 제거했습니다.
- `freeze_image_backbone: false`, `freeze_text_backbone: false` 기본값은 유지했습니다.

#### 왜 바꿨는가

우리는 student text encoder를 DatologyAI retrieval 모델로 쓰기로 이미 결정했습니다. 그런데 설정 파일이 여전히 이전 선택의 흔적을 남기고 있으면, 문서와 실제 동작이 달라집니다.

또 `text_pretrained`는 EVA처럼 모델 이름과 pretrained alias를 따로 넘기는 경우에는 의미가 있지만, `hf-hub:DatologyAI/retr-opt-vit-b-32`처럼 경로 자체가 모델 정체성을 설명하는 경우에는 사실상 필요가 없습니다.

#### 쉽게 설명하면

이 항목은 "기본 student text encoder가 무엇인가"를 정하는 가장 중요한 설정입니다.  
여기가 오래된 값으로 남아 있으면, 사람은 DatologyAI라고 생각하고 학습을 돌리는데 실제로는 다른 모델이 로드될 수 있습니다.

즉 이번 수정은 "우리가 쓰기로 한 모델"과 "실제로 불러오는 모델"을 같게 만들기 위한 정리입니다.

### 4.2 `src/lpcvc_retrieval/data.py`

#### 무엇이 바뀌었는가

- `_resolve_context_length()`를 추가했습니다.
- tokenizer가 가진 실제 `context_length`를 읽어 `OpenClipTokenizerAdapter`가 보관하도록 바꿨습니다.
- train/eval dataset tokenization에서 고정 길이 대신 `self.text_context_length`를 사용하도록 바꿨습니다.
- `train_augment=false`일 때도 설정한 `image_input_size`를 그대로 사용하도록 수정했습니다.

#### 왜 바꿨는가

기존 student 변경 작업에서 가장 쉽게 놓치는 부분이 "텍스트 길이"입니다. 모델 이름은 바꿨는데 데이터 쪽은 예전 길이를 계속 쓰면, 토크나이즈 결과와 모델 입력 기대치가 어긋날 수 있습니다.

이번 DatologyAI 모델은 다행히도 현재 기준 `77` 토큰을 사용합니다. 하지만 코드가 아예 `77`에 박혀 있으면, 이번에는 우연히 맞아도 다음 실험 때 다시 같은 문제를 겪게 됩니다.

또 smoke 테스트 과정에서 `train_augment=false`일 때 train 이미지가 설정 크기와 무관하게 224로 바뀌는 문제를 실제로 확인했습니다. 이건 특히 offline feature extraction이나 재현성 중심 실험에서 바로 문제를 일으킬 수 있는 버그였습니다.

#### 쉽게 설명하면

여기는 "데이터를 모델이 먹을 수 있는 형태로 맞춰주는 곳"입니다.

- text 쪽에서는 "문장을 몇 칸 길이로 자를지"
- image 쪽에서는 "이미지를 몇 픽셀 크기로 맞출지"

를 결정합니다.

즉 이 파일을 고친 이유는, **모델이 기대하는 입력과 데이터가 실제로 들어가는 입력을 같게 만들기 위해서**입니다.

특히 `train_augment=false` 버그는 겉보기에는 작은 문제 같지만, 실제로는 "384로 학습한다고 생각했는데 내부에서는 224로 들어가는" 식의 혼란을 만들 수 있어서 반드시 막아야 했습니다.

### 4.3 `src/lpcvc_retrieval/dual_tower.py`

#### 무엇이 바뀌었는가

- `text_context_length` 속성을 모델 내부에 보관하도록 했습니다.
- `temperature_init`를 받아 `logit_scale` 초기값에 실제로 반영하도록 수정했습니다.

#### 왜 바꿨는가

먼저 `text_context_length`를 모델이 알고 있어야 export나 다른 유틸리티가 "이 모델의 텍스트 입력 길이가 무엇인지"를 안전하게 참조할 수 있습니다.  
그 정보가 모델 밖에만 있으면, data 쪽과 export 쪽이 서로 다른 값을 쓸 위험이 생깁니다.

그리고 `temperature_init`는 config에 이미 있는 값이었지만, 실제 초기화 로직에는 반영되지 않고 있었습니다. 즉 사용자는 설정을 바꾸고 있다고 생각하지만, 실제 모델은 그 값을 무시하고 있었던 셈입니다.

#### 쉽게 설명하면

이 수정은 두 가지를 바로잡은 것입니다.

1. 모델이 자기 텍스트 입력 길이를 스스로 알게 함
2. config에 적어 둔 temperature 값이 진짜로 적용되게 함

즉 "설정 파일에 적힌 내용"과 "모델이 실제로 하는 일"을 일치시키기 위한 수정입니다.

### 4.4 `src/lpcvc_retrieval/model.py`

#### 무엇이 바뀌었는가

- config에서 읽은 `temperature_init`를 `DualTowerStudent` 생성자로 전달하도록 연결했습니다.

#### 왜 바꿨는가

`dual_tower.py`에서 `temperature_init`를 받을 수 있게 바꿔도, 실제 model build 단계에서 그 값을 넘기지 않으면 아무 일도 일어나지 않습니다.

즉 이 수정은 기능 추가라기보다, 이미 정의된 설정값이 실제 모델 생성까지 전달되도록 이어 준 연결 작업입니다.

#### 쉽게 설명하면

설정값이 중간에서 끊기지 않게 배선한 것입니다.  
전구를 바꿔도 전선이 안 이어져 있으면 켜지지 않는 것과 같은 문제였습니다.

### 4.5 `src/lpcvc_retrieval/export.py`

#### 무엇이 바뀌었는가

- `_resolve_text_context_length()`를 추가했습니다.
- ONNX export 시 dummy text input shape를 `model.text_context_length`로 결정하도록 바꿨습니다.
- 관련 설명도 `(1, L_text)` 기준으로 정리했습니다.

#### 왜 바꿨는가

학습이 잘 되어도 export에서 입력 shape를 잘못 잡으면 배포 단계에서 다시 문제가 납니다. 특히 Qualcomm AI Hub에 올릴 때는 input dtype과 shape가 매우 중요합니다.

이번 작업에서는 text ONNX가 `int32[1,77]`로 나가야 했습니다. 이 요구를 안정적으로 만족하려면, export도 모델이 실제로 기대하는 text length를 따라가야 합니다.

#### 쉽게 설명하면

여기는 "훈련한 모델을 배포용 파일로 꺼내는 곳"입니다.  
학습 때는 맞았는데 export만 다른 길이로 내보내면, 나중에 AI Hub나 런타임에서 다시 깨질 수 있습니다.

즉 이 수정은 **훈련용 입력 규칙과 배포용 입력 규칙을 같게 만들기 위한 것**입니다.

### 4.6 `src/lpcvc_retrieval/train.py`

#### 무엇이 바뀌었는가

- EMA 기준으로 best를 판단하는 경우, 실제로 평가에 사용한 weights를 `best.pt`에 저장하도록 수정했습니다.

#### 왜 바꿨는가

기존에는 학습 중에 EMA shadow weights로 성능을 측정하면서도, 막상 `best.pt`에는 raw weights를 저장하고 있었습니다.

이 상태에서는 이런 일이 생깁니다.

1. 학습 중에는 "이번이 best"라고 판단함
2. 그런데 저장된 `best.pt`를 나중에 다시 eval하면, 학습 중 봤던 best 성능과 다를 수 있음

즉 "best checkpoint"라는 이름이 정확하지 않게 됩니다.

#### 쉽게 설명하면

시험 볼 때는 A 답안을 채점해 놓고, 저장은 B 답안을 해 두는 문제였습니다.  
나중에 다시 꺼내 보면 "왜 점수가 다르지?"가 되는 구조였기 때문에 고쳤습니다.

이번 수정으로는 **좋다고 판단한 바로 그 가중치**가 `best.pt`에 저장됩니다.

### 4.7 `scripts/eval.py`, `scripts/export_onnx_split.py`

#### 무엇이 바뀌었는가

- 실행 시 현재 repo의 `src/`를 가장 먼저 import하도록 `sys.path.insert(0, ...)`를 추가했습니다.

#### 왜 바꿨는가

smoke 테스트 중 실제로 확인된 문제입니다. 현재 작업 중인 repo가 아니라, 다른 경로에 설치되어 있던 `lpcvc_retrieval` 패키지가 import되는 상황이 있었습니다.

이 문제는 매우 위험합니다. 이유는 간단합니다.

- 우리는 현재 repo 코드를 고쳤다고 생각함
- 그런데 eval/export는 다른 설치본 코드로 돌아감
- 그러면 "고친 코드가 잘 동작했다"는 결론 자체가 틀릴 수 있음

즉 검증 결과를 신뢰할 수 없게 됩니다.

#### 쉽게 설명하면

수정한 파일을 시험해야 하는데, 시험 감독이 옆 폴더의 옛날 파일을 가져다가 채점하는 상황이었습니다.  
그래서 "지금 이 저장소의 코드"를 확실히 읽도록 경로를 고정했습니다.

## 5. 문서 변경 상세

코드만 바꾸고 문서를 안 바꾸면, 다음 사람이 문서를 믿고 작업하다가 다시 혼란을 겪게 됩니다.  
이번에는 문서도 코드 기준으로 함께 정리했습니다.

### 5.1 `README.md`

#### 무엇이 바뀌었는가

- student text encoder 설명을 DatologyAI 기준으로 갱신했습니다.
- `freeze_*` 설정 의미를 짧고 쉽게 설명했습니다.
- config 설정값이 실제로 어떤 동작을 하는지 빠르게 볼 수 있게 정리했습니다.
- student image/text encoder의 runtime snapshot 표를 추가했습니다.
- validation snapshot과 AI Hub upload-only 결과를 최신 상태로 갱신했습니다.

#### 왜 바꿨는가

README는 가장 먼저 읽는 문서입니다.  
여기에 오래된 모델명이나 불분명한 설정 설명이 남아 있으면, 사용자는 첫 단계부터 잘못 이해하게 됩니다.

특히 `freeze_text_backbone: false`처럼 사용자가 자주 헷갈리는 항목은 README에서도 간단히 설명하는 편이 맞습니다.

#### 쉽게 설명하면

README는 "빠르게 현재 상태를 파악하는 문서"이므로,  
지금 이 저장소의 실제 기본값과 실제 검증 결과가 바로 보여야 합니다.

### 5.2 `docs/PROJECT_GUIDE.md`

#### 무엇이 바뀌었는가

- DatologyAI text tower 설명과 선정 이유를 추가했습니다.
- config 키별 상세 설명을 보강했습니다.
- `freeze_*`, `temperature_init`, `offline_feature_dir`, `teacher_weight_mode` 등 주요 설정의 실제 동작을 자세히 설명했습니다.

#### 왜 바꿨는가

사용자는 결국 "이 설정이 실제로 무슨 일을 하느냐"를 가장 많이 묻게 됩니다.  
GUIDE는 바로 그 질문에 답하는 문서여야 합니다.

그래서 이번에는 README보다 더 자세하게, 실제 코드 기준으로 설명을 넣었습니다.

#### 쉽게 설명하면

README가 빠른 소개라면, GUIDE는 "왜 이 값이 필요한지"까지 설명하는 문서입니다.  
이번 수정은 GUIDE가 진짜 매뉴얼 역할을 하도록 만든 작업입니다.

### 5.3 `docs/PROJECT_MAP.md`

#### 무엇이 바뀌었는가

- student text encoder 기본값을 DatologyAI 기준으로 정리했습니다.
- 검증 상태를 "준비 중"이 아니라 실제 smoke / upload-only 완료 기준으로 갱신했습니다.

#### 왜 바꿨는가

PROJECT_MAP은 현재 프로젝트 상태를 빠르게 복원하는 문서입니다.  
이미 끝난 작업이 계속 "준비 중"으로 적혀 있으면, 다음 작업자가 현재 진척도를 잘못 이해하게 됩니다.

#### 쉽게 설명하면

이 문서는 지도 역할을 하므로, 현재 위치가 틀리면 그 뒤 판단도 다 틀어집니다.  
그래서 완료된 상태를 완료된 상태로 고쳐 둔 것입니다.

## 6. 이번 패치에서 실제로 확인한 동작

이번 작업은 문서 수정으로 끝내지 않았고, 실제 smoke 테스트까지 진행했습니다.

### 테스트 조건

- device: CPU
- student text encoder: `hf-hub:DatologyAI/retr-opt-vit-b-32`
- student image encoder: 현재 student 기본 설정 사용
- smoke teacher: `ViT-B-32 (openai)`, `ViT-B-16 (openai)`
- synthetic dataset: `tmp/smoke_e2e`

### 확인한 흐름

1. online distillation train
2. offline teacher feature extraction
3. offline distillation train
4. eval
5. ONNX split export
6. Qualcomm AI Hub upload-only

### 결과

- online train: 통과
- offline feature extraction: 통과
- offline train: 통과
- eval: 통과
- export: 통과
- text ONNX input: `int32[1,77]` 확인
- AI Hub upload-only: 통과

업로드된 AI Hub model id:

- image encoder: `mn493d1rq`
- text encoder: `mmx26z7kn`

### 이 테스트가 의미하는 것

이 결과는 아래를 확인해 줍니다.

- student text encoder 교체 후에도 전체 파이프라인이 이어진다
- text encoder ONNX 입력이 Qualcomm AI Hub 요구 조건인 `int32`로 나간다
- 최소한의 기능 검증 기준으로는 online / offline / eval / export / upload 흐름이 모두 살아 있다

### 이 테스트가 아직 보장하지 않는 것

- 실제 대형 teacher 조합에서의 GPU 메모리 사용량
- 실제 대형 teacher 조합에서의 학습 속도
- 최종 성능이 얼마나 오르는지

즉 이번 smoke는 "코드가 맞게 연결되어 있는가"를 확인한 테스트이지, 최종 accuracy를 확정하는 실험은 아닙니다.

## 7. 추가로 함께 수정된 문제

이번 패치는 text encoder 교체가 중심이었지만, 테스트 과정에서 드러난 몇 가지 중요한 문제도 같이 수정했습니다.

### 문제 1. EMA best checkpoint 저장 의미 불일치

- 바뀐 점: 평가한 EMA weights를 그대로 `best.pt`에 저장
- 왜 중요한가: 나중에 다시 eval해도 train 중 best 판단과 같은 대상을 보게 됨

### 문제 2. `train_augment=false`일 때 train 이미지 크기 224로 고정

- 바뀐 점: 설정한 `image_input_size`를 그대로 사용
- 왜 중요한가: 384 입력으로 실험한다고 생각했는데 내부에서 224로 바뀌는 혼란을 방지

### 문제 3. eval/export 스크립트가 다른 설치본 패키지를 읽을 수 있음

- 바뀐 점: 현재 repo `src/`를 최우선 import
- 왜 중요한가: 지금 고친 코드로 검증했다는 사실 자체를 신뢰할 수 있게 됨

## 8. 남아 있는 참고 사항

현재 문서에도 적어 둔 것처럼 아래 설정은 아직 "설정은 존재하지만 active train path에서 실사용되지는 않는 상태"입니다.

- `loss.label_smoothing`

이 항목을 이번 패치에서 바로 제거하지 않은 이유는 다음과 같습니다.

- 현재 주 손실 경로가 `siglip_loss` 기준이기 때문
- 당장 runtime 오류를 만드는 문제는 아니기 때문
- 이번 작업의 핵심은 student text encoder 전환과 그에 따른 정합성 확보였기 때문

즉 이 항목은 "당장 고장 난 부분"이 아니라 "차후 정리 대상"에 더 가깝습니다.

## 9. 최종 정리

이번 패치를 한 문장으로 정리하면 이렇습니다.

**student text encoder를 DatologyAI retrieval 모델로 바꾸고, 그 결정이 설정, 데이터 처리, export, 문서, 검증 결과까지 모두 일관되게 반영되도록 저장소 전체를 정리한 작업입니다.**

조금 더 쉽게 말하면:

- 모델 이름만 바꾼 것이 아니라
- 그 모델을 기준으로 데이터가 들어가고
- 그 모델을 기준으로 export가 되고
- 그 모델을 기준으로 문서가 설명되고
- 실제로 학습/평가/업로드 흐름이 끝까지 이어지는지 확인한 것입니다

그래서 이 패치노트는 "무엇을 바꿨는가"만 적은 기록이 아니라,  
"왜 그렇게 바꿔야 했는가"와 "그 결과 무엇이 더 안전해졌는가"까지 설명하는 기록으로 보는 것이 맞습니다.
