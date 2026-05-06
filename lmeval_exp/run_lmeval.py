"""KIVI(진짜 양자화) LM-Eval 평가 스크립트.

논문 KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache (Liu et al., 2024)
Table 3의 KIVI-2 / KIVI-4 행을 재현하기 위한 스크립트.

A-1 (fake_quant_exp/) 와의 차이
-------------------------------
- A-1: LlamaAttention.forward를 monkey-patch하여 fake quantization 적용
- A-2(여기): KIVI 저자가 작성한 LlamaForCausalLM_KIVI 모델 클래스 사용.
            진짜 2bit 저장 + residual fp16 sliding window 포함.

사용 예:
    # 16bit baseline
    python run_lmeval.py --k_bits 16 --v_bits 16 \
        --tasks coqa,truthfulqa_gen,gsm8k --output results/16bit.json

    # KIVI-2 (논문 핵심)
    python run_lmeval.py --k_bits 2 --v_bits 2 \
        --tasks coqa,truthfulqa_gen,gsm8k --output results/KIVI-2.json

    # KIVI-4
    python run_lmeval.py --k_bits 4 --v_bits 4 \
        --tasks coqa,truthfulqa_gen,gsm8k --output results/KIVI-4.json
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import torch
from transformers import LlamaConfig, AutoTokenizer, AutoModelForCausalLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str,
                   default="/mnt/data/gloomyteam/kivi_clone/models/Llama-2-7b-hf",
                   help="모델 경로 (로컬 절대 경로 권장)")
    p.add_argument("--tasks", type=str, default="coqa,truthfulqa_gen,gsm8k",
                   help="lm-eval task 이름들, 콤마로 구분")

    # KIVI 핵심 hyperparameters (논문 default)
    p.add_argument("--k_bits", type=int, default=2,
                   help="K cache 비트 수. 16=baseline, 4=KIVI-4, 2=KIVI-2")
    p.add_argument("--v_bits", type=int, default=2,
                   help="V cache 비트 수")
    p.add_argument("--group_size", type=int, default=32,
                   help="양자화 group size (논문 default: 32)")
    p.add_argument("--residual_length", type=int, default=128,
                   help="fp16으로 보존할 최근 토큰 수 (논문 default: 128)")

    # lm-eval 설정
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_fewshot", type=int, default=None,
                   help="None이면 lm-eval task default 사용 "
                        "(coqa=0, truthfulqa_gen=0, gsm8k=5)")
    p.add_argument("--limit", type=int, default=None,
                   help="task당 평가 샘플 수 제한 (sanity용). None이면 전체.")

    p.add_argument("--output", type=str, default="results.json")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_kivi_or_baseline_model(args):
    """args.k_bits / v_bits 값에 따라 KIVI 모델 또는 baseline을 로드.

    - 16bit: 일반 transformers AutoModelForCausalLM (KIVI 양자화 미적용)
             — A-1 baseline과 동일한 경로
    - 2/4bit: LlamaForCausalLM_KIVI + config.k_bits/v_bits/group_size/residual_length
              + config.use_flash=True (KIVI는 flash-attn 강제)
    """
    if args.k_bits >= 16 and args.v_bits >= 16:
        # ===== Baseline (양자화 없음) =====
        print(f"[load] baseline (16bit) from {args.model}")
        # 16bit에서는 굳이 KIVI 클래스 안 써도 됨 — 결과 동일.
        # eager 사용: A-1 baseline과 정확히 동일 경로로 비교 가능.
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            trust_remote_code=True,
        ).cuda()
        return model

    # ===== KIVI 양자화 적용 =====
    print(
        f"[load] KIVI (K={args.k_bits}bit, V={args.v_bits}bit, "
        f"group={args.group_size}, residual={args.residual_length}) "
        f"from {args.model}"
    )
    # KIVI repo 코드 import — 작업 디렉토리가 kivi_clone일 때 정상 동작
    # KIVI 모듈은 kivi_clone 루트에 있음. cwd가 거기여도 sys.path에
    # 자동으로 안 들어가는 환경이 있어서 명시적으로 추가.
    import sys, os
    KIVI_ROOT = "/mnt/data/gloomyteam/kivi_clone"
    if KIVI_ROOT not in sys.path:
        sys.path.insert(0, KIVI_ROOT)
    from models.llama_kivi import LlamaForCausalLM_KIVI

    config = LlamaConfig.from_pretrained(args.model)
    config.k_bits = args.k_bits
    config.v_bits = args.v_bits
    config.group_size = args.group_size
    config.residual_length = args.residual_length
    config.use_flash = True   # llama_kivi.py 내부 assert가 True 강요

    model = LlamaForCausalLM_KIVI.from_pretrained(
        pretrained_model_name_or_path=args.model,
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    ).cuda()
    return model


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # 1) 모델 로드 (16bit / KIVI-4 / KIVI-2)
    model = load_kivi_or_baseline_model(args)
    model.eval()

    # 2) Tokenizer — KIVI example.py와 동일한 옵션
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=False, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3) lm-eval-harness로 평가
    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("ERROR: lm-eval가 설치되어 있지 않습니다. "
              "(pip install 'lm-eval==0.4.2' --no-deps)", file=sys.stderr)
        sys.exit(1)

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"[eval] tasks={tasks}, num_fewshot={args.num_fewshot}, "
          f"limit={args.limit}")

    # num_fewshot=None이면 lm-eval가 task default 사용 (gsm8k는 5-shot이 default)
    eval_kwargs = dict(model=lm, tasks=tasks, limit=args.limit)
    if args.num_fewshot is not None:
        eval_kwargs["num_fewshot"] = args.num_fewshot

    results = evaluator.simple_evaluate(**eval_kwargs)

    # 4) 출력 + 저장
    config_dict = {
        "model": args.model,
        "k_bits": args.k_bits, "v_bits": args.v_bits,
        "group_size": args.group_size,
        "residual_length": args.residual_length,
        "num_fewshot": args.num_fewshot,
        "limit": args.limit,
    }
    task_results = results.get("results", {})

    print("\n=========== RESULT ===========")
    print(f"config: {config_dict}")
    for task, metrics in task_results.items():
        print(f"\n[{task}]")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:>22} : {v:.4f}")
            else:
                print(f"  {k:>22} : {v}")
    print("================================")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".",
                exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"config": config_dict, "results": task_results}, f,
                  indent=2, default=str)
    print(f"\n[save] {args.output}")


if __name__ == "__main__":
    main()
