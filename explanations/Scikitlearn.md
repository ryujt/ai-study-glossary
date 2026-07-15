# Scikit‑learn (사이킷런)

> **Scikit-learn은 파이썬에서 딥러닝을 뺀 고전 머신러닝을 표준화된 한 가지 사용법으로 묶어 놓은 라이브러리다.**

## 머신러닝계의 공구 세트다

전동 드릴, 렌치, 드라이버를 저마다 다른 방식으로 쥐고 돌려야 한다면 작업은 엉망이 된다. **Scikit-learn은 수십 종의 머신러닝 알고리즘에 똑같은 손잡이를 달아 놓은 공구 세트다.** 회귀든 분류든 군집화든, 도구를 바꿔도 쥐는 법은 항상 같다.

그 손잡이는 세 개의 메서드로 요약된다.

- `fit(X, y)` — 데이터로 모델을 학습시킨다.
- `predict(X)` — 학습한 모델로 새 데이터를 예측한다.
- `transform(X)` — 데이터를 변환한다(전처리·차원 축소 등).

[선형 회귀](Machine_Learning.md)를 [랜덤 포레스트](Random_Forest.md)로 바꾸든 [SVM](Support_Vector_Machine___SVM.md)으로 바꾸든, 코드에서 바뀌는 건 모델을 만드는 한 줄뿐이다. 나머지 `fit`·`predict`는 그대로다. 이 일관성이 Scikit-learn의 핵심 가치다.

## 데이터는 표, 정답은 열로 넣는다

Scikit-learn은 데이터를 **2차원 배열 `X`**(행은 샘플, 열은 [특성](Feature_Engineering.md))와 **정답 벡터 `y`**로 받는다. 이 규약 하나로 [NumPy](NumPy.md) 배열이든 [Pandas](Pandas.md) 데이터프레임이든 그대로 흘려보낼 수 있다.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)          # 학습
pred = model.predict(X_test)         # 예측
print(accuracy_score(y_test, pred))  # 평가
```

전처리·모델을 하나로 묶는 `Pipeline`, 하이퍼파라미터를 자동 탐색하는 [`GridSearchCV`](Grid_Search.md), 신뢰할 만한 성능을 재는 [교차 검증](CrossValidation.md)도 모두 같은 `fit`/`predict` 규약 위에서 돌아간다.

## 흔한 오해

> ⚠️ **흔한 오해 — "Scikit-learn으로 딥러닝도 하면 되겠네."**
>
> Scikit-learn은 신경망 학습을 위한 도구가 아니다. GPU 가속도, 자동 미분도, 복잡한 신경망 구성도 지원하지 않는다. 간단한 `MLPClassifier`가 있긴 하지만 실전용은 아니다. 신경망·딥러닝은 [TensorFlow](TensorFlow.md)나 [PyTorch](PyTorch.md)의 몫이고, Scikit-learn은 그 앞단의 **정형 데이터 머신러닝**에서 빛난다.

## 언제 쓰나

| 상황 | Scikit-learn이 맞나 |
|------|--------------------|
| 표 형태의 정형 데이터로 분류·회귀 | 맞다. 가장 먼저 꺼낼 도구다 |
| [군집화](Clustering.md)·[차원 축소](Dimensionality_Reduction.md) 같은 비지도 작업 | 맞다 |
| 전처리·모델·평가를 한 파이프라인으로 | 맞다 |
| 이미지·음성·자연어 딥러닝, GPU 대규모 학습 | 아니다. TensorFlow·PyTorch로 |

## 정리하면

- Scikit-learn은 고전 머신러닝 알고리즘을 **`fit`/`predict`/`transform`** 이라는 하나의 사용법으로 통일한 라이브러리다.
- 데이터는 특성 행렬 `X`와 정답 `y`로 넣고, 모델만 바꿔 끼우면 나머지 코드는 그대로다.
- [지도학습](Supervised_Learning.md)·[자율학습](Unsupervised_Learning.md), 전처리, 교차 검증, 하이퍼파라미터 탐색까지 한 틀 안에 있다.
- 딥러닝은 담당이 아니다. 정형 데이터 머신러닝의 표준 출발점으로 쓴다.

## 함께 읽기

- [NumPy](NumPy.md) — Scikit-learn이 데이터를 담는 배열의 토대.
- [Pandas](Pandas.md) — 데이터프레임을 그대로 `X`로 흘려보낼 수 있는 짝꿍.
- [Cross-Validation](CrossValidation.md) — Scikit-learn으로 성능을 신뢰성 있게 재는 법.
- [Grid Search](Grid_Search.md) — 같은 틀 위에서 하이퍼파라미터를 자동 탐색.
- [TensorFlow](TensorFlow.md) — Scikit-learn이 손대지 않는 딥러닝을 맡는 프레임워크.
