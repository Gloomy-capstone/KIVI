"""KIVI Table 1 / Table 3 fake quantization 결과 재현 스크립트.

usage:
    python run_eval.py \
        --model meta-llama/Llama-2-7b-hf \
        --tasks coqa,truthfulqa_gen \
        --k_bits 2 --v_bits 2 --k_dim channel --v_dim token \
        --output results/2bit_KC_VT.json

지원 configuration (논문 Table 3, Llama-2-7B 기대 결과 — 참고):

    | config              | k_bits | v_bits | k_dim   | v_dim   | CoQA  | TQA   |
    |---------------------|--------|--------|---------|---------|-------|-------|
    | 16bit               | 16     | 16     | -       | -       | 63.88 | 30.76 |
    | 4bit (K-T, V-T)     | 4      | 4      | token   | token   | 64.82 | 29.85 |
    | 2bit (K-T, V-T)     | 2      | 2      | token   | token   | 39.88 | 18.29 |
    | 2bit (K-C, V-C)     | 2      | 2      | channel | channel |  3.60 |  0.27 |
    | 2bit (K-T, V-C)     | 2      | 2      | token   | channel |  1.30 |  0.49 |
    | 2bit (K-C, V-T)     | 2      | 2      | channel | token   | 59.08 | 33.10 |  ← KIVI 핵심
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llama_patch import patch_llama, set_kv_quant_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf",
                   help="HF model id 또는 로컬 경로")
    p.add_argument("--tasks", type=str, default="coqa,truthfulqa_gen",
                   help="lm-eval task 이름들, 콤마로 구분")
    p.add_argument("--k_bits", type=int, default=16)
    p.add_argument("--v_bits", type=int, default=16)
    p.add_argument("--k_dim", type=str, default="token", choices=["token", "channel"])
    p.add_argument("--v_dim", type=str, default="token", choices=["token", "channel"])
    p.add_argument("--group_size", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_fewshot", type=int, default=0)
    p.add_argument("--limit", type=int, default=None,
                   help="task 당 평가 샘플 수 제한 (디버깅용). None이면 전체.")
    p.add_argument("--output", type=str, default="results.json")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # 1) Fake quant config 설정 + LlamaAttention patching
    set_kv_quant_config(
        k_bits=args.k_bits, v_bits=args.v_bits,
        k_dim=args.k_dim, v_dim=args.v_dim,
        group_size=args.group_size,
    )
    patch_llama()

    # 2) 모델 로딩 — 반드시 eager attention!
    print(f"[load] {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",   # ★ 중요: monkey-patch가 적용되는 경로
        trust_remote_code=True,
    )
    model.eval()

    # 3) lm-eval-harness로 평가
    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("ERROR: lm-eval가 설치되어 있지 않습니다.\n"
              "  pip install 'lm-eval==0.4.2'  (또는 호환 버전)",
              file=sys.stderr)
        sys.exit(1)

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"[eval] tasks={tasks}, num_fewshot={args.num_fewshot}, limit={args.limit}")

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
    )

    # 4) 결과 출력 / 저장
    config = {
        "model": args.model,
        "k_bits": args.k_bits, "v_bits": args.v_bits,
        "k_dim": args.k_dim, "v_dim": args.v_dim,
        "group_size": args.group_size,
        "num_fewshot": args.num_fewshot,
        "limit": args.limit,
    }
    task_results = results.get("results", {})

    print("\n=========== RESULT ===========")
    print(f"config: {config}")
    for task, metrics in task_results.items():
        print(f"\n[{task}]")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:>20} : {v:.4f}")
            else:
                print(f"  {k:>20} : {v}")
    print("================================")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"config": config, "results": task_results}, f, indent=2, default=str)
    print(f"\n[save] {args.output}")


if __name__ == "__main__":
    main()
