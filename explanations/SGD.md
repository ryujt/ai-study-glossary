# SGD (Stochastic Gradient Descent, SGD)

**SGD (Stochastic Gradient Descent, SGD)**

1.  **정의:** SGD는 머신러닝 모델의 파라미터를 조정하여 손실 함수를 최소화하는 최적화 알고리즘입니다. 각 업데이트는 전체 데이터셋 대신 하나의 데이터 샘플 또는 작은 미니배치를 사용하여 수행됩니다.

2.  **핵심 개념:**
    *   **손실 함수 (Loss Function):** 모델의 예측값과 실제값 사이의 차이를 측정하는 함수입니다. SGD는 이 함수를 최소화하는 것을 목표로 합니다.
    *   **파라미터 (Parameters):** 모델 학습 중에 조정되는 변수들입니다. 가중치(weights)와 편향(biases)이 대표적인 예시입니다.
    *   **학습률 (Learning Rate):** 파라미터 업데이트의 크기를 결정하는 하이퍼파라미터입니다. 너무 크면 발산하고, 너무 작으면 학습 속도가 느립니다.
    *   **미니배치 (Mini-Batch):** 전체 데이터셋에서 일부를 선택하여 학습에 사용하는 작은 그룹입니다.

3.  **작동 방식:**
    *   SGD는 모델이 예측한 값과 실제 값의 차이 (손실)를 계산합니다.
    *   각 데이터 샘플(또는 미니배치)에 대해, 이 손실 값은 모델의 파라미터를 업데이트하는 데 사용되는 그래디언트(gradient)를 계산하는 데 사용됩니다.
    *   그래디언트는 손실 함수가 가장 가파르게 증가하는 방향을 나타냅니다.
    *   따라서, 그래디언트의 반대 방향으로 파라미터를 조정하여 손실 함수를 줄입니다.
    *   이 과정을 반복하여 모델을 학습시킵니다.

4.  **응용 분야:**
    *   심층 신경망 (Deep Neural Networks) 학습
    *   선형 회귀 (Linear Regression) 및 로지스틱 회귀 (Logistic Regression)와 같은 간단한 모델 학습
    *   추천 시스템 (Recommender Systems)
    *   이미지 인식 (Image Recognition)

5.  **관련 용어:**
    *   경사 하강법 (Gradient Descent)
    *   미니배치 최적화 (Mini-Batch Optimization)
    *   Adam
    *   Momentum