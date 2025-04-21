# Early Stopping (조기 종료)

**조기 종료 (Early Stopping)**

1.  **정의:** 모델의 학습 과정에서 과적합(Overfitting)을 방지하기 위해, 특정 성능 기준을 만족하는 시점에서 학습을 중단하는 기술입니다.

2.  **핵심 개념:**
    *   **과적합 (Overfitting):** 모델이 학습 데이터에만 지나치게 맞춰져 새로운 데이터에 대한 일반화 성능이 떨어지는 현상.
    *   **검증 데이터 (Validation Data):** 학습에 사용되지 않은 별도의 데이터 세트로 모델의 성능을 평가하는 데 사용됩니다.
    *   **성능 지표 (Performance Metric):** 모델의 성능을 측정하는 지표 (예: 정확도, 정밀도, 재현율, 손실 함수 값).
    *   **Epoch:** 학습 데이터 전체를 한 번 학습하는 것을 의미합니다.

3.  **작동 방식:**
    *   모델의 학습 과정에서 검증 데이터에 대한 성능을 주기적으로 평가합니다.
    *   검증 데이터의 성능이 더 이상 개선되지 않거나, 오히려 성능이 저하되는 경우 학습을 중단합니다.
    *   학습된 모델을 최종적으로 평가하고, 학습률(Learning Rate)을 조정하여 개선할 수 있습니다.

4.  **응용 분야:**
    *   신경망 (Neural Networks) 학습
    *   의사 결정 트리 (Decision Trees) 학습
    *   서포트 벡터 머신 (Support Vector Machines) 학습

5.  **관련 용어:**
    *   하이퍼파라미터 튜닝 (Hyperparameter Tuning)
    *   정규화 (Regularization)
    *   교차 검증 (Cross-Validation)
    *   조기 종료 기준 (Early Stopping Criterion)