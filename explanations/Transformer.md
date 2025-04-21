# Transformer (트랜스포머)

## Transformer (트랜스포머)

**1. 정의:**

Transformer는 순환 신경망(RNN)의 단점을 극복하고, 특히 자연어 처리(NLP) 분야에서 뛰어난 성능을 보이는 딥러닝 모델 아키텍처입니다. 어텐션 메커니즘을 기반으로 문장 전체를 한 번에 처리하여 병렬 처리가 가능합니다.

**2. 핵심 개념:**

*   **어텐션(Attention):** 입력 데이터의 각 부분 간의 관련성을 학습하여, 모델이 중요한 부분에 집중하도록 합니다.
*   **셀프 어텐션(Self-Attention):** 동일한 입력 데이터 내의 단어 간의 관계를 파악합니다.
*   **인코더-디코더 구조(Encoder-Decoder Structure):**  입력 시퀀스를 이해하고, 목표 시퀀스를 생성하는 데 사용되는 구조입니다.
*   **포지셔널 인코딩(Positional Encoding):**  Transformer는 순서 정보를 학습하지 않으므로, 시퀀스 내 단어의 위치 정보를 추가합니다.
*   **병렬 처리(Parallel Processing):** RNN의 순차적인 처리 방식과 달리, Transformer는 전체 입력을 동시에 처리하여 계산 효율성을 높입니다.

**3. 작동 방식:**

Transformer는 크게 인코더와 디코더로 구성됩니다.

*   **인코더:** 입력 문장을 여러 개의 '셀'을 통해 처리합니다. 각 셀은 어텐션 메커니즘을 사용하여 문장 내의 모든 단어 간의 관계를 파악하고, 각 단어의 중요도를 결정합니다.
*   **디코더:** 인코더의 결과를 바탕으로 목표 시퀀스를 생성합니다.  또한, 디코더 역시 어텐션 메커니즘을 사용하여 인코더의 출력을 참고하여 다음 단어를 예측합니다.

**4. 응용 분야:**

*   **자연어 번역(Machine Translation):**  기계 번역 시스템에 널리 사용됩니다.
*   **텍스트 요약(Text Summarization):**  긴 텍스트를 짧게 요약하는 데 사용됩니다.
*   **질의 응답(Question Answering):**  질문과 관련된 정보를 텍스트에서 추출합니다.
*   **텍스트 생성(Text Generation):**  새로운 텍스트를 생성하는 데 사용됩니다. (예: GPT 모델)

**5. 관련 용어:**

*   **RNN (Recurrent Neural Network):** 순차 데이터를 처리하는 신경망 모델.
*   **CNN (Convolutional Neural Network):** 이미지 처리 분야에서 주로 사용되는 신경망 모델.
*   **딥러닝(Deep Learning):**  여러 계층의 인공 신경망을 사용하여 데이터에서 패턴을 학습하는 기법.
*   **BERT (Bidirectional Encoder Representations from Transformers):**  Transformer 아키텍처를 기반으로 한 언어 모델.
*   **GPT (Generative Pre-trained Transformer):**  Transformer 아키텍처를 기반으로 한 생성형 언어 모델.
