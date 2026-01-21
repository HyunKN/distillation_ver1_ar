**MobileCLIP2-Retrieval-Optimization (LPCVC 2026)**

이 프로젝트는 LPCVC 2026 Image-to-Text Retrieval 챌린지를 위한 모바일 최적화 모델 학습 저장소입니다. 기존 베이스라인 모델의 성능을 극대화하기 위해 MobileCLIP2 (Reinforced S2) 전략을 도입하고, 저지연/고효율 환경에서 최적의 Recall@10 성능을 목표로 합니다.

**Key Improvements & Training Strategy**

1. SigLip Loss & Logit Bias 도입

SigLip Loss: 기존 Softmax 기반의 Contrastive Loss 대신 Sigmoid 기반의 Loss를 채택. 특히 배치 사이즈가 작은 환경에서도 학습 안정성을 높이고 메모리 효율을 개선했습니다.Logit Bias: 모델 출력 단계에서 학습 가능한 logit_bias 파라미터를 추가하여, 특정 샘플에 대한 과적합을 방지하고 검색(Retrieval) 성능을 정교화했습니다.

2. Multi-GT Masking 지원
  
(Data Pipeline)src/lpcvc_retrieval/data.py를 수정하여 데이터 로더가 image_id를 반환하도록 변경했습니다.이를 통해 하나의 이미지에 여러 캡션이 매칭되는 Multi-Ground Truth 상황에서 불필요한 Penalty를 줄이는 마스킹 처리가 가능해졌습니다.3. Reinforced S2 Knowledge DistillationMobileCLIP2의 Reinforced S2 전략을 활용하여, 가벼운 모델임에도 불구하고 강력한 이미지-텍스트 정렬(Alignment) 성능을 유지합니다.w_distill_affinity 가중치를 조정하여 Teacher 모델의 지식을 효과적으로 전이받습니다.

**Project StructurePlaintextLPCVC_Demo/**
├── config.yaml               # 하이퍼파라미터 및 학습 환경 설정 (lr: 5e-4, epochs: 30 등)
├── src/
│   └── lpcvc_retrieval/
│       ├── data.py           # Multi-GT 지원 및 image_id 반환 커스텀 데이터 로더
│       ├── model.py          # Logit Bias가 적용된 MobileCLIP2 모델 정의
│       ├── losses.py         # SigLip Loss 등 핵심 손실 함수 구현
│       └── train.py          # 최적화된 학습 루프 및 Evaluation 로직
└── README.md

**Hyperparameters**
현재 최적의 성능을 내기 위한 파라미터 값을 찾아보는 중임.
조만간 학습 결과 공유 예정

How to Train
환경 설정 확인: config.yaml 파일 내의 데이터 경로와 하이퍼파라미터가 적절한지 확인
학습 실행: Bash에서 python src/lpcvc_retrieval/train.py --config config.yaml (python3일 수도 있음)

 **Main Changes Note (Recent Update)**
 train.py: SigLip 연산 최적화 및 체크포인트 저장 로직 강화
 model.py: 모바일 배포를 고려한 아키텍처 경량화 및 Bias 추가
 data.py: image_id 반환 로직 추가로 정답지(GT) 인식 오류 해결
