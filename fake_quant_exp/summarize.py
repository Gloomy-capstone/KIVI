"""results/*.json들을 모아 논문 Table 1/3 형식의 요약표를 출력."""

import glob
import json
import os
import sys


# 우리가 관심 있는 metric 키 (lm-eval-harness 0.4.x 기준)
COQA_KEYS = ["em,none", "em", "exact_match,none", "exact_match"]
TQA_KEYS = ["bleu_max,none", "bleu_max", "bleu_acc,none", "bleu_acc"]


def pick_metric(metrics: dict, candidates):
    for k in candidates:
        if k in metrics and isinstance(metrics[k], (int, float)):
            return k, metrics[k]
    # fallback: 숫자형 metric 중 첫 번째
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and "stderr" not in k:
            return k, v
    return None, None


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    files = sorted(glob.glob(os.path.join(out_dir, "*.json")))
    if not files:
        print(f"No json files in {out_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        cfg = data["config"]
        res = data["results"]

        name = os.path.splitext(os.path.basename(f))[0]
        k_str = f"{cfg['k_bits']}bit/{cfg['k_dim'][0].upper()}"
        v_str = f"{cfg['v_bits']}bit/{cfg['v_dim'][0].upper()}"

        coqa_key, coqa_val = pick_metric(res.get("coqa", {}), COQA_KEYS)
        tqa_key, tqa_val = pick_metric(res.get("truthfulqa_gen", {}), TQA_KEYS)

        rows.append({
            "name": name, "K": k_str, "V": v_str,
            "coqa": coqa_val, "coqa_key": coqa_key,
            "tqa": tqa_val, "tqa_key": tqa_key,
        })

    # print
    print()
    print(f"{'config':<16} {'K':<10} {'V':<10} {'CoQA':>10} {'TruthfulQA':>12}")
    print("-" * 62)
    for r in rows:
        coqa = "-" if r["coqa"] is None else f"{r['coqa'] * 100:>10.2f}"
        tqa = "-" if r["tqa"] is None else f"{r['tqa']:>12.2f}"
        print(f"{r['name']:<16} {r['K']:<10} {r['V']:<10} {coqa} {tqa}")
    print("-" * 62)

    keys_used = set()
    for r in rows:
        if r["coqa_key"]:
            keys_used.add(("coqa", r["coqa_key"]))
        if r["tqa_key"]:
            keys_used.add(("truthfulqa_gen", r["tqa_key"]))
    print(f"metrics used: {sorted(keys_used)}")
    print()
    print("Reference (paper Table 3, Llama-2-7B):")
    print("  16bit                CoQA 63.88 / TruthfulQA 30.76")
    print("  4bit  (K-T,V-T)      CoQA 64.82 / TruthfulQA 29.85")
    print("  2bit  (K-T,V-T)      CoQA 39.88 / TruthfulQA 18.29")
    print("  2bit  (K-C,V-C)      CoQA  3.60 / TruthfulQA  0.27")
    print("  2bit  (K-T,V-C)      CoQA  1.30 / TruthfulQA  0.49")
    print("  2bit  (K-C,V-T)      CoQA 59.08 / TruthfulQA 33.10  <- KIVI 핵심")


if __name__ == "__main__":
    main()
