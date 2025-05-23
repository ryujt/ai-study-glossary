# Batch Normalization (배치 정규화)

## 배치 정규화 (Batch Normalization) 설명

**1. 정의:** 배치 정규화는 딥러닝 모델의 학습 속도를 높이고, 모델의 성능을 향상시키기 위해 각 배치 내의 데이터 샘플들의 분포를 평균 0, 분산 1로 정규화하는 기술입니다.

**2. 핵심 개념:**

*   **평균(Mean):** 배치 내의 각 특징(feature)의 평균값을 계산합니다.
*   **분산(Variance):** 각 특징의 분산값을 계산합니다.
*   **하버드(Huber) 손실 함수:** 학습 시 분산이 큰 경우, 분산이 작은 경우에 대비하여 손실 함수를 조정합니다.
*   **지수적 감쇠(Exponential Moving Average):** 각 배치의 통계량을 안정화하기 위해 활용됩니다.
*   **학습률(Learning Rate):** 배치 정규화의 영향을 고려하여 적절한 학습률을 설정해야 합니다.

**3. 작동 방식:**

1.  **통계량 계산:** 각 배치 내의 각 특징에 대해 평균(μ)과 분산(σ²)을 계산합니다.
2.  **정규화:** 각 데이터 샘플의 값을 평균을 중심으로 표준편차(σ)만큼 정규화합니다.
    `x_normalized = (x - μ) / √(σ² + ε)` (ε는 작은 상수 값으로, 분모가 0이 되는 것을 방지합니다.)
3.  **스케일(Scale) 및  shifted (Shifted):** 정규화된 값에 의해 스케일 파라미터(γ)와 shifted 파라미터(β)를 곱하여 원래의 활성화 함수를 복원합니다.
    `y = γ * x_normalized + β`

**4. 응용 분야:**

*   **합성곱 신경망(CNN):** 이미지 분류, 객체 탐지 등의 작업에서 널리 사용됩니다.
*   **순환 신경망(RNN):** 시퀀스 데이터 처리에서 학습 안정성을 높이는 데 활용됩니다.
*   **Transformer:** 자연어 처리 모델의 성능 향상에 기여합니다.
*   **일반적인 딥러닝 모델:** 대부분의 딥러닝 모델에서 일반적인 정규화 방법으로 사용됩니다.

**5. 관련 용어:**

*   **정규화 (Normalization):** 데이터의 분포를 조정하는 기술의 일반적인 용어입니다.
*   **활성화 함수 (Activation Function):** 딥러닝 모델의 비선형성을 부여하는 함수입니다.
*   **손실 함수 (Loss Function):** 모델의 예측과 실제 값의 차이를 측정하는 함수입니다.
*   **오차 역전파 (Backpropagation):** 딥러닝 모델의 가중치를 업데이트하는 방법입니다.
*   **Leaky ReLU:** 활성화 함수 중 하나로, ReLU의 문제점을 해결하기 위해 사용됩니다.
