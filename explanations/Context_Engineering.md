# 컨텍스트 엔지니어링 (Context Engineering)

컨텍스트 엔지니어링은 AI가 작업할 때 참고할 정보를 고르고, 필요한 시점에 제공하고, 갱신하는 일이다.

## 개발할 때

결제 오류를 수정한다면 관련 함수, 재현 로그, 결제 규칙, 테스트 명령을 제공한다. 조사 중 새 사실이 나오면 반영하고, 무관한 대화나 오래된 로그는 줄인다.

## 기억할 점

자료를 많이 넣는 것만으로 품질이 좋아지지는 않는다. 파일 전체보다 필요한 부분과 경로를 알려 주고, AI가 실제로 어떤 근거를 읽었는지 확인한다.

근거: [Anthropic: 컨텍스트 엔지니어링](https://www.anthropic.com/research/effective-context-engineering-for-ai-agents).

## 함께 읽기

- [저장소 지침](./Repository_Instructions.md) — 반복해서 필요한 프로젝트 정보를 둔다.
- [컨텍스트 압축](./Context_Compaction.md) — 늘어난 기록을 정리한다.
