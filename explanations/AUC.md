# AUC (Area Under the Curve)

## AUC (Area Under the Curve) 설명

**1. 정의:**

AUC(Area Under the Curve)는 ROC(Receiver Operating Characteristic) 곡선 아래의 면적을 의미합니다. ROC 곡선은 모델의 판별 능력(discrimination)을 시각적으로 나타내는 그래프로, 거짓 양성(False Positive)과 진실 양성(True Positive)의 비율을 다양한 임계값에서 보여줍니다. AUC는 이 곡선 아래 면적을 통해 모델의 전체 성능을 하나의 숫자로 나타냅니다.

**2. 핵심 개념:**

*   **ROC 곡선:** 모델의 판별 능력을 시각적으로 표현하는 그래프입니다.
*   **진정한 양성(True Positive, TP):** 실제로 긍정적인 클래스에 대해 모델이 긍정적으로 예측한 경우입니다.
*   **거짓 양성(False Positive, FP):** 실제로 긍정적인 클래스가 아니지만 모델이 긍정적으로 예측한 경우입니다.
*   **임계값(Threshold):** 모델의 예측 확률을 기준으로 실제 긍정/부정으로 분류하는 기준값입니다.
*   **정확도(Precision):** 모델이 긍정적으로 예측한 것 중 실제로 긍정적인 비율을 나타냅니다.

**3. 작동 방식:**

ROC 곡선은 True Positive Rate (TPR, Sensitivity, Recall)과 False Positive Rate (FPR, 1-Specificity)를 그래프로 나타냅니다. TPR은 진실 양성 비율을, FPR은 거짓 양성 비율을 나타냅니다. AUC는 이 두 값의 관계를 나타내는 곡선 아래 면적이 얼마나 넓은지를 나타냅니다. AUC 값이 1에 가까울수록 모델의 성능이 좋고, 0에 가까울수록 성능이 낮습니다.

**4. 응용 분야:**

*   **의료 진단:** 질병 진단 모델의 정확도 평가
*   **스팸 필터링:** 스팸 메일 식별 모델의 성능 평가
*   **신용 사기 탐지:** 사기 거래 탐지 모델의 성능 평가
*   **이미지 분류:** 객체 인식 모델의 성능 평가

**5. 관련 용어:**

*   **정밀도-재현율 곡선 (Precision-Recall Curve):** 정밀도와 재현율을 보여주는 곡선입니다.
*   **F1 점수:** 정밀도와 재현율의 조화 평균입니다.
*   **Confusion Matrix:** 모델의 예측 결과를 실제 값과 비교하여 오류를 분석하는 표입니다.
*   **모델 평가 지표:** 모델의 성능을 측정하고 비교하기 위한 다양한 지표입니다.
*   **학습 곡선(Learning Curve):** 학습 데이터의 양에 따른 모델 성능 변화를 나타내는 그래프입니다.