# KIVI Fake Quantization 재현 실험 (Llama-2-7B)

논문 *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache* (Liu et al., 2024)
의 **Table 1 / Table 3** fake quantization 실험을 Llama-2-7B로 재현한다.

> "Fake" quantization = KV cache를 fp16에서 Q→DQ만 수행하고, 실제 저정밀로 저장하지는
> 않는 방식. 메모리 절감 효과는 없지만 **정확도 영향만 깨끗이 측정**할 수 있어 논문에서
> ablation 도구로 사용된다.

## 파일 구성

```
fake_quant_exp/
├── fake_quant.py     # per-token / per-channel 양자화 함수 (RTN, asymmetric)
├── llama_patch.py    # LlamaAttention.forward를 monkey-patch
├── run_eval.py       # lm-eval-harness로 평가하는 메인 스크립트
├── summarize.py      # results/*.json을 논문 표 형식으로 요약
├── run_all.sh        # 6개 configuration 일괄 실행
└── README.md
```

## 사전 준비

```bash
source /mnt/data/gloomyteam/kivi/bin/activate

# lm-eval-harness 설치 (없다면)
pip install 'lm-eval==0.4.2'
```

모델 경로가 로컬에 있다면 환경변수로 지정:

```bash
export MODEL=/path/to/Llama-2-7b-hf      # default: meta-llama/Llama-2-7b-hf
```

## 실행

### 한 번에 6개 configuration 모두

```bash
cd fake_quant_exp
bash run_all.sh
```

### 개별 configuration

```bash
# 16bit baseline
python run_eval.py --output results/16bit.json

# KIVI의 핵심 — 2bit (K per-channel, V per-token)
python run_eval.py \
    --k_bits 2 --v_bits 2 --k_dim channel --v_dim token \
    --output results/2bit_KC_VT.json

# 빠른 sanity check (50 샘플만)
python run_eval.py \
    --k_bits 2 --v_bits 2 --k_dim channel --v_dim token \
    --limit 50 --output results/sanity.json
```

## 기대 결과 (논문 Table 3, Llama-2-7B)

| config              | K        | V        |  CoQA |  TruthfulQA |
|---------------------|----------|----------|------:|------------:|
| 16bit               | 16/-     | 16/-     | 63.88 |       30.76 |
| 4bit (K-T, V-T)     | 4/token  | 4/token  | 64.82 |       29.85 |
| 2bit (K-T, V-T)     | 2/token  | 2/token  | 39.88 |       18.29 |
| 2bit (K-C, V-C)     | 2/chan   | 2/chan   |  3.60 |        0.27 |
| 2bit (K-T, V-C)     | 2/token  | 2/chan   |  1.30 |        0.49 |
| **2bit (K-C, V-T)** | 2/chan   | 2/token  | **59.08** | **33.10** |

핵심 관찰:
- 2bit V per-channel은 어떤 K 설정과 조합되어도 모두 무너짐 (3.60, 1.30) → **OB2**
- 2bit에서 가장 정확도가 높은 조합은 **K per-channel + V per-token** → **OB3**
- 4bit per-token은 16bit와 거의 차이 없음 → **OB1**

## 구현 디테일

### 1. monkey-patch 위치

`LlamaAttention.forward`에서 KV cache concat 직후, `repeat_kv` 직전에 fake quantization을
끼워 넣는다. 이렇게 하면:

- prefill: 전체 prompt KV가 Q→DQ된 상태로 attention에 들어감
- decoding: 매 step마다 전체 cache가 Q→DQ된 상태로 attention에 들어감

→ 실제로 캐시를 저정밀로 저장한 시스템과 **수치적으로 동치**.

### 2. per-channel zero-padding

decoding 도중에는 token 수가 group_size로 나누어떨어지지 않는 순간이 빈번하다.
이때는 **0으로 padding**해서 그룹 크기를 맞춘 뒤 양자화하고, dequantize 결과에서
padding 부분을 잘라낸다 (논문 Section 3.1과 동일).

### 3. attention implementation

`attn_implementation="eager"`로 모델을 로드해야 한다. SDPA / Flash-Attn 경로는
별도 모듈(`LlamaSdpaAttention`, `LlamaFlashAttention2`)을 쓰기 때문에 patching이
적용되지 않는다.

## 트러블슈팅

- **GPU OOM**: RTX 4000 Ada (20GB)에서는 `--batch_size 1`로 충분. 모델만 ~13GB.
- **lm-eval task 이름이 안 맞음**: 0.3.x에서는 `truthfulqa_gen`, 0.4.x에서도 동일.
  CoQA는 두 버전 모두 `coqa`. 이름이 안 먹히면 `lm_eval --tasks list`로 확인.
- **결과가 너무 다르게 나옴**: 16bit configuration부터 먼저 돌려보고 baseline이
  논문 수치(63.88 / 30.76)와 일치하는지 확인. 다르면 lm-eval 버전 차이 또는
  prompt template 차이일 가능성이 높다.
