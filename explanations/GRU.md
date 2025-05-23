# GRU (Gated Recurrent Unit)

## GRU (Gated Recurrent Unit) 설명

**1. 정의:**

GRU는 순환 신경망(RNN)의 한 종류로, 과거 정보를 효과적으로 기억하고 잊는 데 특화된 재귀 신경망 구조입니다.  RNN의 단점을 보완하기 위해 설계되었으며, ‘게이트’라는 메커니즘을 사용하여 정보의 흐름을 제어합니다.

**2. 핵심 개념:**

*   **게이트 (Gate):** GRU의 핵심 요소로, 정보를 선택적으로 통과시키거나 막는 역할을 합니다. 게이트는 학습된 가중치를 사용하여 정보를 필터링합니다.
*   **업데이트 게이트 (Update Gate):**  과거 정보의 중요도를 결정하고, 현재 상태에 반영하는 데 사용됩니다.
*   **리셋 게이트 (Reset Gate):**  과거 정보를 얼마나 잊을지를 결정합니다.
*   **메모리 (Memory):**  GRU가 과거 정보를 저장하는 공간입니다.

**3. 작동 방식:**

GRU는 각 시점에서 다음과 같은 단계를 거칩니다.

1.  **입력:** 현재 입력 데이터와 이전 시점의 메모리 상태가 입력됩니다.
2.  **업데이트 게이트 계산:**  업데이트 게이트는 현재 입력과 이전 메모리 상태를 기반으로 계산됩니다.
3.  **리셋 게이트 계산:**  리셋 게이트는 현재 입력과 업데이트 게이트를 기반으로 계산됩니다.
4.  **메모리 업데이트:**  메모리는 업데이트 게이트와 리셋 게이트를 사용하여 업데이트됩니다. 즉, 새로운 정보를 추가하거나 기존 정보를 수정합니다.
5.  **출력 생성:**  메모리와 현재 입력이 결합되어 출력으로 사용됩니다.

**4. 응용 분야:**

*   **자연어 처리 (NLP):** 텍스트 생성, 기계 번역, 감성 분석 등 다양한 NLP 작업에 활용됩니다.
*   **시계열 예측:** 주가 예측, 날씨 예측 등 시계열 데이터 분석에 사용됩니다.
*   **음성 인식:** 음성 데이터를 텍스트로 변환하는 데 사용됩니다.
*   **기계 번역:** 한 언어를 다른 언어로 번역하는 데 활용됩니다.

**5. 관련 용어:**

*   **RNN (Recurrent Neural Network):** 순환 신경망, 시간 순서대로 데이터를 처리하는 신경망.
*   **LSTM (Long Short-Term Memory):** GRU와 유사하지만, 더 복잡한 게이트 구조를 사용하는 RNN의 한 종류입니다.
*   **순환 신경망 (RNN):**  RNN은 이전 시점의 정보를 기억하고 이를 사용하여 현재 시점의 출력을 예측하는 신경망의 한 종류입니다.
*   **가역 신경망 (FNN):**  순환 신경망과 달리, 각 시점의 출력이 이전 시점의 입력에만 의존하는 신경망입니다.
*   **배치 정규화 (Batch Normalization):**  학습 속도를 높이고 모델의 안정성을 향상시키는 기술입니다.