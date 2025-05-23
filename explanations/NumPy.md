# NumPy (넘파이)

## NumPy (넘파이)

**1. 정의:** NumPy는 파이썬에서 과학적 계산을 위한 핵심 라이브러리입니다. 다차원 배열 객체와 이러한 배열에서 작동하는 고수학 함수들을 제공합니다.

**2. 핵심 개념:**

*   **다차원 배열 (ndarray):**  NumPy의 가장 중요한 개념입니다. 다양한 데이터 타입(정수, 실수 등)을 하나의 배열에 저장하며, 행(row)과 열(column)을 가진 다차원 구조를 지원합니다.
*   **브로드캐스팅 (Broadcasting):** NumPy는 서로 다른 모양의 배열 간에도 연산을 수행할 수 있도록 브로드캐스팅이라는 기능을 제공합니다.  더 작은 배열을 더 큰 배열의 모양에 맞게 자동으로 확장하여 연산을 수행합니다.
*   **벡터화 (Vectorization):**  NumPy는 루프를 사용하지 않고 배열의 모든 요소에 동시에 적용할 수 있는 벡터화된 연산을 제공합니다. 이는 파이썬 루프보다 훨씬 빠르고 효율적입니다.
*   **메모리 효율성:** NumPy 배열은 파이썬 리스트보다 메모리를 효율적으로 사용합니다.  데이터를 연속적인 메모리 공간에 저장하여 속도 향상에 기여합니다.
*   **선형대수, 푸리에 변환, 난수 생성:** NumPy는 이러한 고급 수학 연산을 위한 함수들을 포함합니다.

**3. 작동 방식:**

NumPy는 C로 구현되어 있어 파이썬보다 훨씬 빠릅니다.  NumPy 배열은 메모리 내에서 연속적으로 저장되며, C로 작성된 함수들이 이러한 배열에 직접 작동합니다.  NumPy는 배열의 인덱싱, 슬라이싱, 브로드캐스팅, 벡터화 연산을 통해 효율적인 데이터 처리를 가능하게 합니다.  NumPy 배열의 연산은 기본적으로 배열 요소 간의 연산을 수행하며, 브로드캐스팅을 통해 다양한 모양의 배열 간의 연산이 가능합니다.

**4. 응용 분야:**

*   **데이터 분석 및 과학 계산:**  데이터 시각화, 통계 분석, 머신러닝 모델 훈련 등에서 핵심적인 역할을 합니다.
*   **이미지 처리:**  이미지를 NumPy 배열로 표현하고, 이미지 필터링, 특징 추출 등 다양한 이미지 처리 작업을 수행합니다.
*   **금융 모델링:**  주가 예측, 리스크 관리 등 금융 모델링에 활용됩니다.
*   **물리학 및 공학 시뮬레이션:**  수치적 해법을 통해 복잡한 물리 시스템을 시뮬레이션합니다.

**5. 관련 용어:**

*   **Pandas:** NumPy를 기반으로 데이터 분석을 위한 라이브러리입니다.  데이터프레임이라는 데이터 구조를 제공하여 데이터 조작 및 분석을 쉽게 합니다.
*   **Scikit-learn:** 머신러닝 모델을 구축하고 평가하기 위한 라이브러리입니다.  NumPy 배열을 사용하여 머신러닝 모델을 훈련하고 예측합니다.
*   **TensorFlow/PyTorch:** 딥러닝 프레임워크로서, 텐서라는 다차원 배열 객체를 사용하여 딥러닝 모델을 구축하고 훈련합니다.
*   **Matplotlib/Seaborn:**  데이터 시각화를 위한 라이브러리입니다. NumPy 배열의 데이터를 기반으로 다양한 그래프와 차트 등을 생성합니다.
*   **GPU 컴퓨팅:**  GPU(Graphics Processing Unit)를 사용하여 NumPy 배열 연산을 가속화하는 기술입니다.  대규모 데이터 처리에 효과적입니다.