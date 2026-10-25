# AI 활용 개발 용어집

AI로 코드를 작성하고, 버그를 고치고, 테스트와 리뷰를 진행하는 개발자를 위한 한국어 용어집이다. 모델을 직접 학습하거나 AI 핵심 기술을 구현하는 지식 대신, 개발 도구를 이해하고 결과를 검증하는 데 필요한 46개 용어를 다룬다.

## 대상 독자

기본적인 코드 작성과 Git 사용을 배우는 신입 개발자부터 AI 코딩 도구를 업무에 쓰는 개발자까지 읽을 수 있다. 머신러닝 지식이나 수학은 필요하지 않다. 각 글은 짧은 정의, 개발 예시, 기억할 점, 관련 용어로 구성한다.

## 읽는 순서

필요한 용어부터 찾아 읽어도 된다. 처음이라면 아래 순서를 권한다.

1. [AI 보조 개발](./explanations/AI_Assisted_Development.md) → [LLM](./explanations/Large_Language_Model___LLM.md) → [환각](./explanations/Hallucination.md): 무엇을 맡길 수 있고 무엇을 확인해야 하는지 이해한다.
2. [프롬프트](./explanations/Prompt.md) → [컨텍스트 엔지니어링](./explanations/Context_Engineering.md) → [저장소 지침](./explanations/Repository_Instructions.md): 작업과 프로젝트 정보를 전달한다.
3. [에이전트](./explanations/Agent.md) → [하네스](./explanations/Agent_Harness.md) → [루프 엔지니어링](./explanations/Loop_Engineering.md): 작업이 실행되고 반복되는 구조를 이해한다.
4. [완료 조건](./explanations/Acceptance_Criteria.md) → [AI 생성 코드 리뷰](./explanations/Code_Review.md) → [가드레일](./explanations/Guardrails.md): 결과와 권한을 관리한다.

## 헷갈리기 쉬운 네 가지 엔지니어링

| 용어 | 다루는 질문 | 개발 예시 |
| --- | --- | --- |
| [프롬프트 엔지니어링](./explanations/Prompt_Engineering.md) | 어떤 요청을 할까? | 재현 절차와 기대 동작을 요청에 적는다. |
| [컨텍스트 엔지니어링](./explanations/Context_Engineering.md) | 어떤 정보를 언제 줄까? | 관련 코드와 로그를 골라 제공한다. |
| [하네스 엔지니어링](./explanations/Harness_Engineering.md) | 어떤 환경과 규칙에서 실행할까? | 도구·검사 명령·권한 설정을 개선한다. |
| [루프 엔지니어링](./explanations/Loop_Engineering.md) | 어떻게 작업을 이어 가고 끝낼까? | 작업 선택·실행·검증·기록·재시도·종료를 연결한다. |

서로 겹치는 개념이다. 특히 루프 엔지니어링은 최근 실무에서 쓰이는 표현으로, 고정된 표준 분류는 아니다. [Addy Osmani의 설명](https://addyosmani.com/blog/loop-engineering/)을 참고하되, 이 용어집에서는 개발자가 적용할 수 있는 작업 방식에 초점을 둔다.

## 목차

### 1. AI 도구를 이해하는 기본 용어

모델의 응답과 한계를 이해하는 데 필요한 개념이다.

- [대규모 언어 모델 (LLM)](./explanations/Large_Language_Model___LLM.md) — 코드와 설명을 생성하는 모델의 역할을 이해한다.
- [생성형 AI (Generative AI)](./explanations/Generative_AI.md) — AI가 만드는 결과물을 개발의 출발점으로 활용한다.
- [토큰과 토큰화 (Token & Tokenization)](./explanations/Tokenization.md) — 입력과 출력의 양을 세는 단위를 이해한다.
- [컨텍스트 윈도 (Context Window)](./explanations/Context_Window.md) — 한 번에 다룰 수 있는 정보량의 한계를 이해한다.
- [환각 (Hallucination)](./explanations/Hallucination.md) — 그럴듯한 오류를 알아보고 근거로 확인한다.
- [멀티모달 AI (Multimodal AI)](./explanations/Multimodal_AI.md) — 화면 캡처 등 여러 종류의 입력을 활용한다.
- [추론 모델 (Reasoning Model)](./explanations/Reasoning_Model.md) — 복잡한 문제 풀이와 사용량의 관계를 이해한다.

### 2. 요청과 프로젝트 맥락

원하는 작업과 필요한 정보를 AI에 전달하는 방법이다.

- [프롬프트 (Prompt)](./explanations/Prompt.md) — 목표와 조건을 구체적으로 전달한다.
- [프롬프트 엔지니어링 (Prompt Engineering)](./explanations/Prompt_Engineering.md) — 요청을 다듬고 결과가 나아지는지 확인한다.
- [컨텍스트 엔지니어링 (Context Engineering)](./explanations/Context_Engineering.md) — 필요한 정보를 골라 적절한 시점에 제공한다.
- [퓨샷 프롬프팅 (Few-shot Prompting)](./explanations/Fewshot_Learning.md) — 예시로 원하는 결과의 패턴을 보여 준다.
- [컨텍스트 압축 (Context Compaction)](./explanations/Context_Compaction.md) — 긴 기록을 줄일 때 보존할 정보를 구분한다.
- [에이전트 메모리 (Agent Memory)](./explanations/Agent_Memory.md) — 다음 작업에 필요한 정보를 대화 밖에 남긴다.
- [저장소 지침 (Repository Instructions)](./explanations/Repository_Instructions.md) — 프로젝트 규칙과 작업 명령을 문서로 공유한다.
- [명세 (Specification)](./explanations/Specification.md) — 구현 전에 요구와 예외를 정리한다.

### 3. 에이전트와 반복 작업

작업 환경을 마련하고 여러 단계를 이어 가는 방법이다.

- [AI 보조 개발 (AI-assisted Development)](./explanations/AI_Assisted_Development.md) — 조사부터 리뷰까지 AI와 협업하는 방식을 익힌다.
- [바이브 코딩 (Vibe Coding)](./explanations/Vibe_Coding.md) — 코드 검토를 생략하는 개발 방식의 의미와 한계를 이해한다.
- [에이전트 (Agent)](./explanations/Agent.md) — AI가 도구를 사용해 여러 단계를 수행하는 방식을 이해한다.
- [에이전트 루프 (Agent Loop)](./explanations/Agent_Loop.md) — 행동과 결과 확인이 반복되는 구조를 이해한다.
- [에이전트 하네스 (Agent Harness)](./explanations/Agent_Harness.md) — 모델의 작업을 가능하게 하는 주변 환경을 이해한다.
- [하네스 엔지니어링 (Harness Engineering)](./explanations/Harness_Engineering.md) — 도구·규칙·검사를 개선해 반복되는 실패를 줄인다.
- [루프 엔지니어링 (Loop Engineering)](./explanations/Loop_Engineering.md) — 작업 선택부터 검증·기록·종료까지 반복 체계를 설계한다.
- [워크플로 (Workflow)](./explanations/Workflow.md) — 개발 단계를 정하고 단계마다 AI를 활용한다.
- [서브에이전트 (Subagent)](./explanations/Subagent.md) — 하위 작업을 나눠 맡길 때의 역할과 한계를 이해한다.
- [에이전트 스킬 (Agent Skills)](./explanations/Agent_Skills.md) — 특정 작업의 절차와 자료를 재사용한다.
- [훅 (Hooks)](./explanations/Hooks.md) — 정해진 시점의 처리와 검사를 자동화한다.
- [워크트리 (Git Worktree)](./explanations/Worktree.md) — 동시 작업의 파일 공간을 분리한다.

### 4. 도구 연결과 자료 검색

AI가 프로젝트 밖의 기능과 근거 자료를 활용하는 방식이다.

- [도구 사용 (Tool Use / Function Calling)](./explanations/Tool_Use.md) — 모델의 요청이 실제 명령 실행으로 이어지는 과정을 이해한다.
- [MCP (Model Context Protocol)](./explanations/MCP.md) — 외부 도구와 자료를 연결하는 공통 규약을 이해한다.
- [검색 증강 생성 (RAG)](./explanations/Retrieval_Augmented_Generation___RAG.md) — 관련 자료를 찾아 답변의 근거로 활용한다.
- [임베딩 (Embedding)](./explanations/Embedding.md) — 의미가 비슷한 자료를 찾는 검색 방식을 이해한다.
- [벡터 데이터베이스 (Vector Database)](./explanations/Vector_Database.md) — 임베딩을 저장하고 검색하는 역할을 이해한다.

### 5. 검증과 권한 관리

완료를 확인하고 맡길 수 있는 범위를 정하는 기준이다.

- [완료 조건 (Acceptance Criteria)](./explanations/Acceptance_Criteria.md) — 완료를 확인할 수 있는 기준을 적는다.
- [평가 (Evaluation / Evals)](./explanations/Evaluation.md) — 같은 사례와 기준으로 작업 결과를 비교한다.
- [AI 생성 코드 리뷰 (Reviewing AI-generated Code)](./explanations/Code_Review.md) — 생성된 코드의 의도와 동작을 검토한다.
- [사람의 개입 (Human-in-the-loop)](./explanations/Human_in_the_Loop.md) — 사람이 확인하거나 결정할 지점을 정한다.
- [가드레일 (Guardrails)](./explanations/Guardrails.md) — 행동과 결과에 검사와 제한을 둔다.
- [샌드박스 (Sandbox)](./explanations/Sandbox.md) — 명령이 접근할 수 있는 실행 범위를 제한한다.
- [프롬프트 인젝션 (Prompt Injection)](./explanations/Prompt_Injection.md) — 참고 자료에 섞인 악의적인 지시를 경계한다.
- [민감정보 보호 (Sensitive Data Protection)](./explanations/Privacy_Preservation.md) — AI가 접근하거나 전송하는 민감정보를 관리한다.
- [체크포인트 (Checkpoint)](./explanations/Checkpoint.md) — 잘못된 변경을 복구할 기준점을 남긴다.

### 6. 모델과 사용량

작업에 맞는 모델을 고르고 사용량을 관리하는 개념이다.

- [모델 선택 (Model Selection)](./explanations/Model_Selection.md) — 실제 작업의 품질·시간·비용으로 모델을 고른다.
- [토큰 예산 (Token Budget)](./explanations/Token_Budget.md) — 반복 실행까지 고려해 사용량을 계획한다.
- [호출 제한 (Rate Limit)](./explanations/Rate_Limit.md) — 요청 수와 처리량의 제한을 이해한다.
- [프롬프트 캐싱 (Prompt Caching)](./explanations/Prompt_Caching.md) — 반복 입력의 처리를 재사용하는 방식을 이해한다.
- [구조화된 출력 (Structured Output)](./explanations/Structured_Output.md) — 후속 도구가 읽을 수 있는 형식으로 결과를 받는다.

## 범위와 근거

신경망 구조, 학습 알고리즘, 모델 훈련 지표, 학습용 프레임워크의 상세 해설은 다루지 않는다. 검색에 쓰는 임베딩과 RAG도 도구 사용자가 알아야 할 수준으로 설명한다.

자료 확인 기준일은 **2026-10-26**이다. 바뀔 수 있는 제품 동작과 최근 용어는 해당 글에 공식 문서나 원문 링크를 남겼다. 예시는 특정 제품의 기능을 보장하는 설명이 아니며, 지원 기능·권한·요금은 사용하는 도구의 문서에서 확인한다.

## 기여와 검증

용어를 추가하거나 수정할 때는 [집필 원칙](./AUTHORING.md)을 따른다. 목차와 관련 용어 링크도 함께 갱신한 뒤 아래 명령으로 문서 구조와 내부 링크를 확인한다. Python 3 표준 라이브러리만 사용한다.

```bash
python3 scripts/check_content.py
```

## 라이선스

[MIT 라이선스](./LICENSE)를 따른다.
