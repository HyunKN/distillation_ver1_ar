# 루트 문서 정리 요약

**날짜**: 2026-03-09  
**상태**: 완료

## 수행한 작업

- 활성 기술 가이드를 루트에서 `docs/PROJECT_GUIDE.md`로 이동
- handover 문서를 `docs/archive/completed-tasks/260228_COMPARISON_AND_TRAINING_METHODS_HANDOVER.md`로 이동
- `README.md`의 문서 링크를 `docs/` 구조에 맞게 갱신
- 활성, 보관 문서 인덱스로 `docs/README.md` 추가
- 생성 산출물 디렉터리인 `features/`를 `.gitignore`에 추가

## 정리 후 구조

```text
docs/
├── README.md
├── PROJECT_GUIDE.md
└── archive/
    ├── ROOT_DOCS_CLEANUP_SUMMARY.md
    └── completed-tasks/
        └── 260228_COMPARISON_AND_TRAINING_METHODS_HANDOVER.md
```

## 정리 이유

- 루트에는 실행 시작점과 프로젝트 개요만 남기기 위해
- 활성 기술 문서를 한 위치에서 찾기 쉽게 만들기 위해
- 완료된 handover와 비교 메모가 루트를 어지럽히지 않도록 하기 위해
- 생성 feature를 소스 파일과 구분하기 위해

## 기능 영향

- 런타임 코드 경로는 바뀌지 않음
- 엔트리 스크립트, import, config 로딩 위치는 유지
- 변경된 것은 문서 경로와 ignore 규칙뿐임