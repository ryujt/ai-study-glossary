# Pandas (판다스)

> **Pandas는 표 형태의 데이터를 읽고 자르고 합치고 요약하는, 파이썬의 데이터 분석 라이브러리다. 코드로 다루는 엑셀이라 생각하면 가깝다.**

## 코드로 조작하는 엑셀 시트다

데이터 분석의 8할은 표를 만지는 일이다. 결측치를 메우고, 조건에 맞는 행을 고르고, 그룹별로 평균을 내는 작업 말이다. 엑셀로도 되지만 수백만 행이거나 같은 처리를 매번 반복해야 하면 손이 아니라 코드가 필요하다. **Pandas는 이 표 조작을 코드로, 그것도 빠르게 해 준다.** 내부는 [NumPy](NumPy.md) 위에 세워져 있어 속도가 받쳐 준다.

자료구조는 두 가지다.

- **Series** — 이름표(인덱스)가 붙은 1차원 배열. 표의 열 하나에 해당한다.
- **DataFrame** — 행과 열로 된 2차원 표. Series 여러 개를 열로 묶은 것이다. Pandas 작업의 중심이다.

## 읽고 → 고르고 → 묶어 요약하는 흐름이다

실제 분석은 대개 "파일을 읽어 → 원하는 행·열을 고르고 → 기준별로 묶어 집계"하는 흐름을 탄다.

```python
import pandas as pd

# 1) 읽기 — CSV를 DataFrame으로
df = pd.read_csv("sales.csv")

# 2) 살펴보기
df.head()               # 앞 5행
df.info()               # 열별 타입과 결측치

# 3) 고르기(필터링) — 조건에 맞는 행만
big = df[df["amount"] > 1000]

# 4) 정제 — 결측치 채우기
df["amount"] = df["amount"].fillna(0)

# 5) 그룹화 집계 — 지역별 매출 합계
by_region = df.groupby("region")["amount"].sum()
```

`groupby`는 "같은 값끼리 묶어 → 각 묶음을 집계"하는 Pandas의 핵심 패턴이다. SQL의 `GROUP BY`와 발상이 같다. CSV뿐 아니라 Excel·SQL·JSON 등 다양한 형식을 `read_*` 함수로 읽어 들일 수 있다.

> 💡 **참고 — 머신러닝의 첫 단추가 대개 Pandas다.** 모델에 넣기 전 데이터를 정제하고 특성을 만드는 [데이터 전처리](Data_Preprocessing.md)와 [특성 공학](Feature_Engineering.md)의 상당 부분이 Pandas로 이뤄진다. 잘 다듬은 DataFrame을 [Scikit-learn](Scikitlearn.md)에 그대로 넘기는 흐름이 흔하다.

## 흔한 오해

> ⚠️ **흔한 오해 — "Pandas는 데이터가 아무리 커도 괜찮다."**
>
> Pandas는 데이터를 통째로 **메모리에 올려** 다룬다. 그래서 메모리보다 큰 데이터에는 버겁다. 그런 규모에는 Dask나 Polars, Spark 같은 도구가 따로 있다. Pandas는 메모리에 들어오는 중소 규모 데이터에서 가장 빛난다.

## 정리하면

- Pandas는 **표 형태 데이터를 코드로 다루는** 파이썬 라이브러리다.
- 1차원 **Series**와 2차원 **DataFrame**이 핵심 자료구조다.
- **읽기 → 필터링 → 정제 → groupby 집계**가 전형적 작업 흐름이다.
- NumPy 위에 서 있고, 머신러닝 전처리의 출발점으로 널리 쓰인다.

## 함께 읽기

- [NumPy](NumPy.md) — Pandas의 속도를 떠받치는 기반 라이브러리.
- [Data Preprocessing](Data_Preprocessing.md) — Pandas가 주로 쓰이는 데이터 정제 단계.
- [Feature Engineering](Feature_Engineering.md) — DataFrame으로 새 특성을 만드는 작업.
- [Scikit-learn](Scikitlearn.md) — 정제한 DataFrame을 넘겨받는 머신러닝 라이브러리.
