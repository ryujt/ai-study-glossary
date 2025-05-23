# Regularization (정규화)

## Regularization (정규화) 설명

**1. 정의:**

Regularization은 머신러닝 모델의 과적합(Overfitting)을 방지하기 위해 모델 복잡도를 제한하는 기술입니다. 모델이 학습 데이터에 지나치게 적합하여 새로운 데이터에 대한 일반화 성능이 떨어지는 현상을 막는 것을 목표로 합니다.

**2. 핵심 개념:**

*   **과적합 (Overfitting):** 모델이 학습 데이터의 노이즈까지 학습하여, 새로운 데이터에 대한 예측 성능이 떨어지는 현상.
*   **일반화 (Generalization):** 모델이 학습하지 않은 새로운 데이터에 대해서도 얼마나 정확하게 예측하는가.
*   **모델 복잡도 (Model Complexity):** 모델의 파라미터 수, 레이어 수 등 모델의 구조적인 복잡도를 의미합니다.
*   **정규화 항 (Regularization Term):**  모델의 손실 함수(Loss Function)에 추가되는 항으로, 모델 복잡도를 줄이는 효과를 가집니다.
*   **L1 정규화 (Lasso Regularization):**  모델의 가중치(Weight)의 절대값 합을 손실 함수에 추가하여, 일부 가중치를 0으로 만들어 특징 선택(Feature Selection) 효과를 제공합니다.

**3. 작동 방식:**

정규화는 모델의 손실 함수에 정규화 항을 추가하여 모델의 복잡도를 제한합니다. 이 정규화 항은 모델이 지나치게 복잡한 패턴을 학습하는 것을 억제하고, 더 단순하고 일반적인 패턴을 학습하도록 유도합니다.  손실 함수를 최소화하는 과정에서 모델의 가중치가 더 작은 값으로 조정되어 복잡도가 줄어듭니다.

**4. 응용 분야:**

*   **신경망 (Neural Networks):**  과적합 방지에 널리 사용됩니다.
*   **선형 회귀 (Linear Regression):**  가중치에 L1 또는 L2 정규화를 적용하여 모델의 안정성을 높입니다.
*   **Support Vector Machines (SVM):**  모델의 복잡도를 조절하여 성능을 개선합니다.
*   **분류 및 회귀 문제 전반:**  다양한 머신러닝 문제에서 모델의 일반화 성능 향상을 위해 사용됩니다.

**5. 관련 용어:**

*   **손실 함수 (Loss Function):** 모델의 예측 값과 실제 값의 차이를 나타내는 함수.
*   **파라미터 (Parameter):** 모델 학습 과정에서 조정되는 변수 (가중치, 편향 등).
*   **최적화 (Optimization):**  손실 함수를 최소화하기 위한 알고리즘.
*   **학습률 (Learning Rate):**  최적화 알고리즘이 각 단계에서 얼마나 이동할지를 결정하는 값.
*   **검증 데이터 (Validation Data):** 모델의 하이퍼파라미터를 튜닝하거나, 과적합 여부를 확인하는 데 사용되는 데이터.