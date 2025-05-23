# Epoch (에포크)

## Epoch (에포크)

**1. 정의:**

Epoch는 머신러닝 모델을 훈련시키는 데 사용되는 한 번의 전체 데이터셋 학습 단위를 의미합니다. 즉, 모델이 전체 훈련 데이터셋을 한 번 완전히 학습하는 것을 한 에포크라고 합니다.

**2. 핵심 개념:**

*   **훈련 데이터셋:** 모델이 학습하기 위해 사용되는 데이터의 총집합입니다.
*   **반복 (Iteration):** 한 에포크 내에서 모델이 훈련 데이터셋의 각 샘플에 대해 업데이트를 수행하는 과정입니다.
*   **손실 함수 (Loss Function):** 모델의 예측과 실제 값 사이의 오차를 측정하는 함수입니다.
*   **옵티마이저 (Optimizer):** 손실 함수를 기반으로 모델의 파라미터를 조정하여 손실을 최소화하는 알고리즘입니다.

**3. 작동 방식:**

1.  **데이터 준비:** 훈련 데이터셋을 에포크 단위로 나누어 준비합니다.
2.  **모델 예측:** 모델은 훈련 데이터셋의 각 샘플에 대해 예측을 수행합니다.
3.  **손실 계산:** 모델의 예측 결과와 실제 값 사이의 손실이 계산됩니다.
4.  **파라미터 업데이트:** 옵티마이저를 사용하여 모델의 파라미터를 업데이트하여 손실을 줄입니다.
5.  **반복:** 위 과정을 훈련 데이터셋 전체를 학습할 때까지 반복합니다.  각 에포크가 끝나면 모델의 성능이 향상되는지 확인합니다.

**4. 응용 분야:**

*   **심층 신경망 훈련:** 특히 심층 신경망의 경우, 에포크 수가 모델의 성능에 큰 영향을 미칩니다.  너무 적은 에포크는 과소적합(underfitting)을 유발하고, 너무 많은 에포크는 과대적합(overfitting)을 유발할 수 있습니다.
*   **이미지 분류, 자연어 처리:** 이미지 인식, 텍스트 분류 등 다양한 머신러닝 작업에 적용됩니다.
*   **모델 성능 최적화:** 적절한 에포크 수를 결정하여 모델의 성능을 최적화합니다.

**5. 관련 용어:**

*   **과적합 (Overfitting):** 모델이 훈련 데이터에 너무 잘 맞춰져 새로운 데이터에 대한 일반화 성능이 떨어지는 현상입니다.
*   **과소적합 (Underfitting):** 모델이 훈련 데이터의 복잡성을 충분히 학습하지 못하여 성능이 낮은 현상입니다.
*   **배치 크기 (Batch Size):** 한 번의 파라미터 업데이트에 사용되는 데이터 샘플의 수를 의미합니다.
*   **학습률 (Learning Rate):** 파라미터 업데이트의 크기를 결정하는 값입니다.
*   **검증 데이터셋 (Validation Dataset):** 훈련 데이터셋이 아닌 별도의 데이터셋으로 모델의 성능을 평가하는 데 사용됩니다.