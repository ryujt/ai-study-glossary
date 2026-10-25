# 평가 (Evaluation / Evals)

평가는 AI가 맡은 일을 얼마나 잘 수행하는지 정해 둔 사례와 기준으로 확인하는 과정이다.

## 개발할 때

팀의 지침을 바꿨다면 자주 맡기는 버그 수정과 테스트 작성 사례를 다시 수행해 본다. 요구 충족 여부, 사람이 고친 부분, 실행 시간과 사용량을 같은 기준으로 비교한다.

## 기억할 점

한 번 잘된 결과만으로 일반화하기 어렵다. AI가 결과를 채점하는 LLM-as-a-judge도 사용할 수 있지만 판단이 틀릴 수 있다. 실행 검사와 사람의 검토를 함께 활용한다.

근거: [Anthropic: 에이전트 평가](https://www.anthropic.com/research/demystifying-evals-for-ai-agents).

## 함께 읽기

- [완료 조건](./Acceptance_Criteria.md) — 개별 작업의 성공 기준을 세운다.
- [하네스 엔지니어링](./Harness_Engineering.md) — 환경을 바꾼 효과를 확인한다.
