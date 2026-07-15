# MLflow (MLflow)

> **MLflow는 머신러닝 실험 기록부터 모델 버전 관리·배포까지, 모델의 한살이를 한 도구로 추적·관리하는 오픈소스 플랫폼이다.**

## 실험 노트와 모델 창고를 합쳐 놓은 도구다

모델을 만들다 보면 금방 이런 상황에 빠진다. "정확도 0.91 나왔던 그 모델, 학습률이 얼마였더라? 어느 데이터로 돌렸지? 파일이 `model_final_v3_real.pkl`이었나?" 손으로 스프레드시트에 적던 실험 기록은 금세 엉킨다. **MLflow는 이 뒤죽박죽을 대신 정리해 주는, 실험 노트와 모델 창고를 합친 도구다.**

MLflow는 크게 네 조각으로 나뉜다.

| 구성 요소 | 하는 일 |
|-----------|---------|
| **Tracking** | 파라미터·지표·산출물을 실행(run) 단위로 기록하고 비교 ([실험 추적](Experiment_Tracking.md)) |
| **Models** | 모델을 프레임워크 독립적인 표준 포맷으로 저장 |
| **Model Registry** | 저장된 모델에 버전·단계(Staging/Production)를 붙여 중앙 관리 |
| **Projects** | 실행 환경과 명령을 정의해 재현 가능하게 패키징 |

## 코드 몇 줄로 실험이 기록으로 남는다

핵심은 `mlflow.start_run()`으로 실행을 열고, 그 안에서 파라미터·지표·모델을 로그로 남기는 것이다. 나중에 MLflow UI에서 실행들을 나란히 놓고 비교한다.

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

with mlflow.start_run():
    n_estimators = 100
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    mlflow.log_param("n_estimators", n_estimators)   # 하이퍼파라미터 기록
    mlflow.log_metric("accuracy", acc)               # 성능 지표 기록
    mlflow.sklearn.log_model(model, "model")         # 모델 자체를 산출물로 저장
```

저장한 모델은 명령 한 줄로 REST API 서버로 띄울 수 있다.

```bash
mlflow models serve -m runs:/<run_id>/model -p 5000
```

> 💡 **참고 — MLflow는 학습 프레임워크가 아니다.** 모델을 만드는 것은 [Scikit-learn](Scikitlearn.md), [PyTorch](PyTorch.md), [TensorFlow](TensorFlow.md) 같은 라이브러리의 몫이다. MLflow는 그 위에서 "무엇을 어떻게 돌렸고 결과가 어땠는지"를 붙잡아 두는 역할만 한다.

## 흔한 오해

> ⚠️ **흔한 오해 — "MLflow가 곧 MLOps다."**
>
> MLflow는 [MLOps](MLOps.md)를 실현하는 여러 도구 중 하나일 뿐이다. 실험 추적과 모델 레지스트리라는 조각을 담당하지만, CI/CD 파이프라인·인프라 자동화·운영 모니터링까지 통째로 해 주지는 않는다. MLOps는 방법론이고, MLflow는 그 방법론을 떠받치는 도구다.

## 정리하면

- MLflow는 **실험 기록·모델 저장·버전 관리·배포**를 하나로 묶은 오픈소스 플랫폼이다.
- Tracking·Models·Model Registry·Projects 네 조각으로 이루어진다.
- 학습 자체가 아니라 **실험과 모델의 이력을 관리**하는 도구다.
- `log_param`·`log_metric`·`log_model` 몇 줄로 실험이 비교 가능한 기록으로 남는다.

## 함께 읽기

- [Experiment Tracking](Experiment_Tracking.md) — MLflow Tracking이 구현하는 바로 그 개념.
- [MLOps](MLOps.md) — MLflow가 한 조각으로 들어가는 더 큰 그림.
- [Model Serving](Model_Serving.md) — MLflow가 저장한 모델을 실제 서비스로 내보내는 단계.
- [Model Monitoring](Model_Monitoring.md) — 배포한 모델을 계속 지켜보는 다음 단계.
