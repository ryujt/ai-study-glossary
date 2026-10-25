# 검색 증강 생성 (RAG)

검색 증강 생성(retrieval-augmented generation)은 질문과 관련된 자료를 먼저 찾아, 그 자료를 참고해 답을 만드는 방식이다.

## 개발할 때

사내 API 사용법을 물으면 관련 문서 일부를 검색해 모델에 함께 전달한다. 코드 도우미가 저장소의 관련 파일을 찾아 설명에 활용하는 상황도 같은 관점으로 이해할 수 있다.

## 기억할 점

자료를 참고시키는 것이며 모델 자체를 재학습하는 것은 아니다. 답을 확인 가능한 자료의 근거에 연결하는 것을 그라운딩(grounding)이라고 한다. 검색에서 빠진 내용이나 오래된 자료 때문에 답이 틀릴 수 있으므로, 자료의 버전과 실제 인용 부분을 확인한다.

근거: [Google Cloud: RAG 개요](https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/rag-engine/rag-overview), [그라운딩 개요](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/overview).

## 함께 읽기

- [임베딩](./Embedding.md) — 의미가 비슷한 자료를 찾는 방법을 이해한다.
- [환각](./Hallucination.md) — 자료를 제공해도 남는 오류를 살펴본다.
