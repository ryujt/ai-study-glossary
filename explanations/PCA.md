# PCA (주성분 분석)

## 주성분 분석 (PCA) 설명

**1. 정의:**

주성분 분석(PCA)은 데이터의 분산을 최대한 보존하면서 데이터의 차원을 축소하는 통계적 기법입니다. 즉, 중요한 정보를 유지하면서 데이터의 복잡성을 줄이는 것을 목표로 합니다.

**2. 핵심 개념:**

*   **분산(Variance):** 데이터가 평균으로부터 얼마나 퍼져 있는지를 나타내는 척도입니다. PCA는 분산이 큰 성분을 우선적으로 선택합니다.
*   **고차원 데이터:** 여러 변수를 가진 데이터 집합입니다. 예를 들어, 사람의 키, 몸무게, 나이 등이 하나의 변수가 될 수 있습니다.
*   **차원 축소:** 데이터의 변수 수를 줄이는 과정입니다.
*   **직교성(Orthogonality):** PCA에서 얻은 주성분들은 서로 직교합니다. 이는 각 성분이 서로 독립적임을 의미하며, 데이터의 각 차원이 서로 영향을 주지 않도록 합니다.
*   **공분산 행렬(Covariance Matrix):** 변수 간의 관계를 나타내는 행렬입니다. PCA는 공분산 행렬을 사용하여 주성분을 계산합니다.

**3. 작동 방식:**

1.  **데이터 표준화:** 데이터를 평균이 0이고 표준편차가 1인 형태로 변환합니다. (데이터의 척도 차이를 없애기 위해)
2.  **공분산 행렬 계산:** 표준화된 데이터의 공분산 행렬을 계산합니다.
3.  **고유값 및 고유벡터 계산:** 공분산 행렬의 고유값과 고유벡터를 계산합니다.  고유벡터는 각 주성분의 방향을 나타내고, 고유값은 각 주성분이 데이터의 분산을 얼마나 설명하는지를 나타냅니다.
4.  **주성분 선택:** 고유값의 크기를 기준으로 주성분을 선택합니다. 고유값이 큰 순서대로 선택하며, 원하는 차원 수만큼 선택합니다.
5.  **데이터 차원 축소:** 선택된 주성분에 따라 데이터를 재구성하여 차원을 축소합니다.

**4. 응용 분야:**

*   **이미지 처리:** 이미지 압축, 얼굴 인식
*   **생물 정보학:** 유전자 발현 데이터 분석, 단백질 구조 분석
*   **금융:** 신용 평가, 주가 예측
*   **마케팅:** 고객 세분화, 상품 추천
*   **데이터 시각화:** 고차원 데이터를 2차원 또는 3차원으로 축소하여 시각적으로 표현

**5. 관련 용어:**

*   **신뢰 영역(Confidence Interval):** 데이터가 속할 가능성이 있는 범위
*   **데이터 클러스터링(Data Clustering):** 유사한 데이터 포인트를 그룹화하는 기법
*   **행렬 분해(Matrix Decomposition):** 행렬을 더 간단한 형태의 행렬로 분해하는 기법
*   **신뢰성(Reliability):** 데이터의 정확성과 일관성
*   **공분산(Covariance):** 두 변수 간의 상관 관계를 나타내는 척도