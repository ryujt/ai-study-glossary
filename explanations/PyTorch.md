# PyTorch (파이토치)

> **PyTorch는 텐서 연산과 자동 미분을 파이썬답게 제공하는 딥러닝 프레임워크로, 코드를 짜는 대로 실행되는 "동적 그래프" 덕에 연구·실험에서 사실상 표준이 됐다.**

## 넘파이에 GPU와 자동 미분을 붙인 것

[넘파이](NumPy.md)를 써봤다면 PyTorch의 절반은 이미 아는 셈이다. PyTorch의 텐서는 넘파이 배열과 거의 똑같이 다루되, **GPU에서 돌아가고 기울기를 자동으로 계산**해 준다. 딥러닝에서 제일 귀찮은 [역전파](Backpropagation.md) 미분을 `loss.backward()` 한 줄이 대신해 준다.

가장 큰 특징은 **동적 그래프(define-by-run)**다. 그래프를 미리 선언하지 않고, 파이썬 코드가 실행되는 순서대로 계산 그래프가 그려진다. 그래서 디버깅이 평범한 파이썬처럼 쉽다.

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x          # 순전파
y.backward()                # 자동 미분
print(x.grad)               # dy/dx = 2x + 3 = 7.0
```

## 흔한 오해

> ⚠️ **흔한 오해 — "요즘도 PyTorch는 정적 그래프라 느리다."**
>
> 초창기 [TensorFlow](TensorFlow.md) 1.x의 정적 그래프와 헷갈린 것이다. PyTorch는 처음부터 동적 그래프였고, 지금은 `torch.compile`로 정적 최적화까지 얹어 속도도 챙긴다. 연구 논문 구현이 PyTorch로 쏠린 이유가 이 유연함이다.

## 정리하면

- PyTorch는 **GPU 텐서 + 자동 미분**을 파이썬답게 제공한다.
- 코드 실행 순서대로 그래프가 그려지는 **동적 그래프**가 강점이다.
- 넘파이를 알면 진입 장벽이 낮고, 연구·실험의 사실상 표준이다.

## 함께 읽기

- [TensorFlow](TensorFlow.md) — 대표적인 경쟁 프레임워크.
- [NumPy](NumPy.md) — PyTorch 텐서의 사고방식이 그대로 온 곳.
- [Backpropagation](Backpropagation.md) — PyTorch가 자동으로 처리해 주는 미분 과정.
- [Keras](Keras.md) — 더 높은 추상화를 제공하는 상위 API.
