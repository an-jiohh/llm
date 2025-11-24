# Lotto-GPT

작은 GPT 모델을 직접 구현하고, 정제된 합성 데이터를 활용해 “로직 기반 문제(로또 계산)”를 얼마나 수행할 수 있는지 실험한 프로젝트입니다.

이 프로젝트는 HuggingFace 모델을 사용하는 것이 아니라, Transformer 기반 GPT 구조를 직접 구현하고 평가까지를 목표로 진행하였습니다.

자세한 내용은 아래의 블로그에서 참고하실 수 있습니다.

[GPT 1 - 특정 기능 수행 모델까지의 여정](https://an-jiohh.github.io/blog/LLMgpt1)  
[GPT 2 - 로또 번호 자판기 모델 구현, 학습, 결과](https://an-jiohh.github.io/blog/LLMgpt2)
[GPT 3 - 모델 구현 후기](https://an-jiohh.github.io/blog/LLMgpt3) 

## 프로젝트 목적

가설

**잘 정제된 데이터라면, 작은 모델로도 계산에 가까운 문제를 일정 수준까지 수행할 수 있을 것이다.**

이를 위해 “로또 미션” 문제를 LLM 학습용 데이터 형태로 재설계하여 합성 데이터로 학습을 진행했습니다.

## 폴더 구조

```
├── app_cli/ # cli 실행 폴더
│   ├── app.py
│   ├── lotto_gpt_best.pt
│   ├── model_util.py
│   └── requirements.txt
├── app_server/ # server(fastapi) 실행 폴더
│   ├── app.py
│   ├── Dockerfile
│   ├── lotto_gpt_best.pt
│   ├── model_util.py
│   ├── requirements.txt
│   ├── static/
│   │   ├── script.js
│   │   └── style.css
│   └── templates/
│       └── index.html
├── lotto_llm/ # 초기 lotto_llm 구현 폴더
│   ├── dataset_creater.ipynb
│   ├── lotto_gpt_best.pt
│   └── lotto_pretrained_model.ipynb
├── lotto_llm_improve/ # 완성 lotto_llm 구현 폴더
│   ├── dataset_creater.ipynb
│   ├── loss_curve.png
│   ├── lotto_gpt_best.pt
│   └── lotto_pretrained_model.ipynb
├── .gitignore
├── pretrained_model.ipynb # 기본적인 gpt 구현 프로젝트
└── README.md
```

## 실행 방법

### app-cli
```
cd app_cli
pip install -r requirements.txt
python app.py
```
cli 입력 형태로 실행

### app-server
```
cd app_server
docker build -t lotto-llm .
docker run --name lotto-llm -p 8000:8000 lotto-llm
```
docker 형태로 server실행 후 localhost:8080/ "/"으로 접속

## 데이터 셋 포맷

```
<IN>
구입금액={money}
당첨번호={w1,w2,w3,w4,w5,w6}
보너스번호={bonus}
</IN>
###
<OUT>
티켓수={ticket_count}
구매번호:
{ticket1}
{ticket2}
...
3개일치={count3}
4개일치={count4}
5개일치={count5}
5개보너스일치={count5b}
6개일치={count6}
수익률={rate}%
</OUT>
```

10k에서 150k까지 증가하며 학습

## 모델 아키텍처

경량 GPT 구조
- Layers: 4
- Hidden size: 256
- Attention heads: 4
- Max sequence length: 512
- Framework: PyTorch

기본 CrossEntropyLoss 에 아래 전략을 추가 적용했습니다.
- Padding 영역 -100 처리
- 결과 텍스트 영역에 roi_mask 적용
- ROI 영역 토큰에 가중치(alpha) 부여
- Gradient Accumulation 적용

## 학습 환경
- GPU: RTX 4060
- PyTorch
- CUDA

## 평가

커스텀 평가 방식 사용

#### 1. 형식/파싱 레벨
**1-1. 티켓수 형식**
- 티켓수 = N 이 존재하는지
- N이 정수로 파싱되는지

**1-2. 구매번호 줄 형식**
- 구매번호: 라인이 존재하는지
- 그 아래에 [...] 형태의 줄들이 있는지
- 각 줄에서 숫자를 파싱할 수 있는지

**1-3. 라벨(결과) 줄 형식**
- 아래 항목이 모두 존재하고 숫자로 파싱되는지
    - 3개일치 (...) = X
    - 4개일치 (...) = Y
    - 5개일치 (...) = Z
    - 5개보너스일치 (...) = A
    - 6개일치 (...) = B
    - 수익률 = R%

#### 2. 내부 일관성 레벨

**2-1. 티켓수 일관성**
- 티켓수 = 실제 구매번호 줄 개수가 같은지
    
**2-2. 구매번호 자체 유효성**
- 각 줄에 숫자가 정확히 6개인지
- 한 줄 안에 중복 숫자가 없는지
- 각 숫자가 1~45 범위인지

#### 3. 계산 정확도 레벨 (기능 테스트)

LLM 출력에 적힌 “결과 값들”이, 그 LLM이 출력한 “구매번호”로 실제 계산했을 때 맞는지.

**3-1. n개 일치 수 계산 검증**  
    - 구매번호 기반으로 다시 3/4/5/5b/6개 일치 개수를 계산
    - 그 값이 LLM이 쓴 X, Y, Z, A, B와 각각 일치하는지

**3-2. 수익률 계산 검증**  
    - 이 값이 LLM이 쓴 수익률 = 실제 수익율

## 평가 결과

#### 평가 결과 요약

| 항목 | 개수 | 설명 |
|------|------|------|
| n_total | 5000 | 전체 테스트 샘플 수 |
| n_ok | 2298 | 정상적으로 동작한 샘플 (성공) |
| n_fail | 2702 | 실패한 샘플 (실패) |


#### 실패 유형 분포

| 항목 | 개수 | 설명 |
|------|------|------|
| n_format_fail | 293 | 출력 형식 오류 (형식 표현 실패) |
| n_logic_fail | 2409 | 로직 계산 실패 (기능 구현 실패) |

#### 세부 에러 유형

| 항목 | 개수 | 설명 |
|------|------|------|
| ticket_count_parse_error | 0 | 티켓 수 파싱 실패 |
| ticket_lines_format_error | 2 | 티켓 라인 형식 오류 |
| result_labels_parse_error | 11 | 결과 라벨 파싱 실패 |
| ticket_count_mismatch | 35 | 티켓 수 불일치 |
| ticket_numbers_invalid | 245 | 잘못된 번호 형식 |
| match_stats_mismatch | 2408 | 일치 개수 계산 오류 |
| roi_mismatch | 17 | 수익률 계산 오류 |
| test_fail | 0 | 기타 테스트 실패 |

#### 간단 요약

| 지표 | 값 |
|------|------|
| 성공률 | **45.96%** |
| 실패율 | **54.04%** |
| 형식 오류 비중 | **5.86%** |
| 로직 실패 비중 | **48.18%** |
