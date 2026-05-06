# A-2: KIVI Real Quantization LM-Eval 실험 (Llama-2-7B)

논문 *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache* (Liu et al., 2024)
의 **Table 3 KIVI-2 / KIVI-4 / 16bit 행**을 Llama-2-7B로 재현한다.

> A-1 (`fake_quant_exp/`)와의 차이 — A-1은 Q→DQ만 하는 fake quantization을
> monkey-patch로 적용했지만, A-2는 KIVI 저자가 작성한 `LlamaForCausalLM_KIVI`를
> 그대로 사용한다. **진짜 2bit 저장 + 최근 토큰 fp16 보존(residual sliding
> window)** 까지 포함된 실제 KIVI 알고리즘이다.

## 파일 구성

```
lmeval_exp/
├── run_lmeval.py        # 메인 평가 스크립트 (KIVI 모델 로딩 + lm-eval-harness)
├── run_all_lmeval.sh    # 16bit / KIVI-4 / KIVI-2 일괄 실행
├── summarize_lmeval.py  # 결과를 논문 Table 3 형식으로 요약
└── README.md
```

## 사전 준비

이미 A-1 환경이 그대로 살아 있어서 추가 설치는 필요 없다. 확인만:

```bash
source /mnt/data/gloomyteam/kivi/bin/activate
cd /mnt/data/gloomyteam/kivi_clone

python -c "
import torch, transformers, flash_attn, lm_eval
from quant.new_pack import triton_quantize_and_pack_along_last_dim
from quant.matmul import cuda_bmm_fA_qB_outer
from models.llama_kivi import LlamaForCausalLM_KIVI
print('all OK')
"
```

## 실행

### Sanity check (50 샘플)

먼저 16bit + KIVI-2 각각을 50샘플로만 빠르게 확인:

```bash
cd /mnt/data/gloomyteam/kivi_clone   # KIVI 모듈 import 위해 필수

mkdir -p lmeval_exp/results

# 16bit baseline (KIVI 모델 미사용 경로)
python lmeval_exp/run_lmeval.py \
    --k_bits 16 --v_bits 16 \
    --tasks coqa,truthfulqa_gen,gsm8k \
    --limit 50 \
    --output lmeval_exp/results/sanity_16bit.json \
    2>&1 | tee lmeval_exp/results/sanity_16bit.log

# KIVI-2 (양자화 적용 경로 검증)
python lmeval_exp/run_lmeval.py \
    --k_bits 2 --v_bits 2 \
    --tasks coqa,truthfulqa_gen,gsm8k \
    --limit 50 \
    --output lmeval_exp/results/sanity_KIVI-2.json \
    2>&1 | tee lmeval_exp/results/sanity_KIVI-2.log
```

### 본 실험 (전체 dataset)

```bash
cd /mnt/data/gloomyteam/kivi_clone

# 백그라운드 실행 — A-1 때 검증된 패턴
nohup bash lmeval_exp/run_all_lmeval.sh \
    > lmeval_exp/run_all_lmeval.log 2>&1 &
echo "PID: $!"
```

3개 config × 3 task = 9개 평가. A-1 fake quant보다 빠를 가능성이 큼:
- 16bit baseline은 일반 transformers 경로 사용
- KIVI-2/4는 모델이 정상 동작하므로 generation이 EOS에서 빠르게 종료
- A-1에서 망가진 케이스가 8배 느렸던 문제 없음

총 시간 예상: **3~6시간** 정도.

### 모니터링

```bash
pgrep -af run_all_lmeval.sh
tail -f /mnt/data/gloomyteam/kivi_clone/lmeval_exp/run_all_lmeval.log
ls /mnt/data/gloomyteam/kivi_clone/lmeval_exp/results/*.json
```

## 결과 요약

```bash
cd /mnt/data/gloomyteam/kivi_clone
python lmeval_exp/summarize_lmeval.py lmeval_exp/results
```

논문 Table 3 형식으로 출력된다.

## 기대 결과 (논문 Table 3, Llama-2-7B)

| config | K | V | CoQA | TruthfulQA (BLEU) | GSM8K |
|--------|---|---|-----:|------------------:|------:|
| 16bit  | 16/-    | 16/-    | 63.88 | 30.76 | 13.50 |
| KIVI-4 | 4bit    | 4bit    | 63.78 | 30.80 | 13.80 |
| KIVI-2 | 2bit    | 2bit    | 63.05 | 33.95 | 12.74 |

핵심 관찰:
- **KIVI-2가 16bit과 거의 동일한 성능** — 단 2bit로 양자화했는데도 정확도 손실 1% 미만
- A-1의 2bit (K-C, V-T) 53.17과 비교하면 **residual sliding window가 정확도 회복에 결정적**
  임을 알 수 있음 (53.17 → 63.05, 약 +10pt 회복)

## 구현 디테일

### KIVI 모델 로드 패턴

KIVI 저자의 `example.py`/`run_ablation_eval.py`와 동일:

```python
from models.llama_kivi import LlamaForCausalLM_KIVI
from transformers import LlamaConfig

config = LlamaConfig.from_pretrained(MODEL_PATH)
config.k_bits = 2          # 또는 4
config.v_bits = 2          # 또는 4
config.group_size = 32
config.residual_length = 128
config.use_flash = True    # KIVI 내부 assert가 강제

model = LlamaForCausalLM_KIVI.from_pretrained(
    MODEL_PATH, config=config,
    low_cpu_mem_usage=True, torch_dtype=torch.float16,
).cuda()
```

### 16bit baseline 처리

`k_bits >= 16`일 땐 KIVI 모델을 안 쓰고 일반 `AutoModelForCausalLM` (eager
attention)으로 로드. 이렇게 하면:
- A-1 baseline과 정확히 같은 경로 → CoQA 63.88 같은 수치가 양쪽에서 동일하게 나와
  비교 가능성 확보
- KIVI 클래스의 flash_attn 의존을 baseline에서 우회

### lm-eval 0.4.2 → KIVI 모델 호환성

`HFLM(pretrained=model, ...)`은 huggingface 호환 모델 객체를 그대로 받음.
`LlamaForCausalLM_KIVI`는 `LlamaPreTrainedModel` 상속이라 huggingface 인터페이스
(`generate`, `forward` 등)를 그대로 만족. 별도 래퍼 불필요.

## 트러블슈팅

- **`assert getattr(config, "use_flash", False)` 에러** → `config.use_flash = True`
  설정 누락. `run_lmeval.py`에 이미 들어있으니 기본 사용 시 문제없음.
- **`undefined symbol` 등 quant.* import 에러** → CUDA kernel이 다른 PyTorch
  버전용으로 빌드된 것. 현재 환경(torch 2.1.2+cu121)에서는 정상 동작 확인됨.
- **GSM8K few-shot 차이** → lm-eval 0.4.2의 `gsm8k` task default는 5-shot.
  논문도 default 사용으로 추정. `--num_fewshot N`으로 명시적 지정 가능.
- **KIVI 모델 로드 후 OOM** → KIVI는 fp16 모델 위에 양자화 메타데이터가 더 붙어서
  순간 메모리 사용량이 baseline보다 약간 큼. 20GB GPU에서 batch_size 1로는 충분.
