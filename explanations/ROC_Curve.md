# ROC Curve (ROC 곡선)

**ROC Curve (ROC 곡선)**

1.  **정의:** ROC 곡선은 이진 분류 모델의 성능을 평가하는 데 사용되는 그래프입니다. True Positive Rate (TPR)과 False Positive Rate (FPR)을 기준으로 만들어지는 곡선으로, 모델의 분류 능력을 직관적으로 보여줍니다.

2.  **핵심 개념:**
    *   **True Positive Rate (TPR) / Sensitivity / Recall:** 실제 양성 샘플을 얼마나 정확하게 긍정으로 예측했는지를 나타냅니다.
    *   **False Positive Rate (FPR):** 실제 음성 샘플을 긍정으로 잘못 예측하는 비율입니다.
    *   **Threshold (임계값):** 모델의 예측 확률 값을 기준으로 샘플을 긍정 또는 음성으로 분류하는 기준값입니다.
    *   **AUC (Area Under the Curve):** ROC 곡선 아래 면적으로, 모델의 전체적인 분류 성능을 나타냅니다. AUC 값이 클수록 모델의 성능이 좋습니다.
    *   **Random Baseline:** 무작위로 예측하는 것과 비교하여 모델의 성능을 평가합니다.

3.  **작동 방식:**
    *   모델은 주어진 데이터에 대해 각 샘플에 대해 긍정 또는 음성으로 예측합니다.
    *   예측 확률 값을 기준으로 임계값을 설정합니다.
    *   임계값을 사용하여 샘플을 분류하고 True Positive (TP), False Positive (FP), True Negative (TN), False Negative (FN)를 계산합니다.
    *   TPR과 FPR을 계산하고, 이를 x축과 y축으로 하는 그래프를 그립니다. TPR은 0과 1 사이의 값으로, FPR은 0과 1 사이의 값으로 표시됩니다.

4.  **응용 분야:**
    *   **의료 진단:** 암 진단, 감염병 예측 등
    *   **사기 탐지:** 신용 카드 사기, 보험 사기 등
    *   **스팸 필터링:** 이메일 스팸 필터링
    *   **고객 이탈 예측:** 고객의 이탈 가능성을 예측

5.  **관련 용어:**
    *   **Classification (분류):** 데이터를 미리 정의된 범주로 나누는 작업입니다.
    *   **Regression (회귀):** 연속적인 값으로 예측하는 작업입니다.
    *   **Precision (정밀도):** 예측된 긍정 샘플 중 실제 긍정 샘플의 비율입니다.
    *   **F1-Score:** Precision과 Recall의 조화 평균입니다.
    *   **Confusion Matrix (혼동 행렬):** 모델의 예측 결과와 실제 결과 사이의 관계를 보여주는 표입니다.