"""results/*.json들을 모아 논문 Table 3 (KIVI-2 / KIVI-4 행) 형식의 요약을 출력."""

import glob
import json
import os
import sys


# lm-eval 0.4.2 metric 키 (우선순위 순)
COQA_KEYS = ["em,none", "em", "exact_match,none", "exact_match"]
TQA_KEYS = ["bleu_max,none", "bleu_max", "bleu_acc,none", "bleu_acc"]
GSM8K_KEYS = [
    "exact_match,strict-match", "exact_match,flexible-extract",
    "exact_match,none", "exact_match",
]


def pick_metric(metrics: dict, candidates):
    """후보 키 순서대로 찾아 첫 번째 매칭 metric 반환."""
    for k in candidates:
        if k in metrics and isinstance(metrics[k], (int, float)):
            return k, metrics[k]
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and "stderr" not in k:
            return k, v
    return None, None


def fmt(val, scale=100):
    if val is None:
        return "-"
    return f"{val * scale:>8.2f}"


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    files = sorted(glob.glob(os.path.join(out_dir, "*.json")))
    if not files:
        print(f"No json files in {out_dir}", file=sys.stderr)
        sys.exit(1)

    # 표시 순서 — 16bit, KIVI-4, KIVI-2
    order_pref = {"16bit": 0, "KIVI-4": 1, "KIVI-2": 2}

    rows = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        cfg = data["config"]
        res = data["results"]

        name = os.path.splitext(os.path.basename(f))[0]
        coqa_key, coqa_val = pick_metric(res.get("coqa", {}), COQA_KEYS)
        tqa_key, tqa_val = pick_metric(res.get("truthfulqa_gen", {}), TQA_KEYS)
        gsm_key, gsm_val = pick_metric(res.get("gsm8k", {}), GSM8K_KEYS)

        rows.append({
            "name": name,
            "k_bits": cfg["k_bits"], "v_bits": cfg["v_bits"],
            "group": cfg.get("group_size"),
            "residual": cfg.get("residual_length"),
            "coqa": coqa_val, "coqa_key": coqa_key,
            "tqa": tqa_val, "tqa_key": tqa_key,
            "gsm": gsm_val, "gsm_key": gsm_key,
        })

    rows.sort(key=lambda r: order_pref.get(r["name"], 99))

    print()
    print(f"{'config':<12} {'K':>4} {'V':>4} {'group':>6} {'resid':>6} "
          f"{'CoQA':>10} {'TruthQA':>10} {'GSM8K':>10}")
    print("-" * 76)
    for r in rows:
        # CoQA, GSM8K는 0~1 → ×100. TruthfulQA bleu_max는 이미 0~100.
        coqa = fmt(r["coqa"], scale=100)
        gsm = fmt(r["gsm"], scale=100)
        if r["tqa_key"] and "bleu_max" in r["tqa_key"]:
            tqa = fmt(r["tqa"], scale=1)
        else:
            tqa = fmt(r["tqa"], scale=100)

        print(f"{r['name']:<12} {r['k_bits']:>4} {r['v_bits']:>4} "
              f"{str(r['group']):>6} {str(r['residual']):>6} "
              f"{coqa} {tqa} {gsm}")
    print("-" * 76)

    keys_used = set()
    for r in rows:
        for task_name, key in [("coqa", r["coqa_key"]),
                               ("truthfulqa_gen", r["tqa_key"]),
                               ("gsm8k", r["gsm_key"])]:
            if key:
                keys_used.add((task_name, key))
    print(f"metrics used: {sorted(keys_used)}")
    print()
    print("Reference (paper Table 3, Llama-2-7B):")
    print("  16bit                CoQA 63.88 / TruthfulQA 30.76 / GSM8K 13.50")
    print("  KIVI-4               CoQA 63.78 / TruthfulQA 30.80 / GSM8K 13.80")
    print("  KIVI-2               CoQA 63.05 / TruthfulQA 33.95 / GSM8K 12.74")


if __name__ == "__main__":
    main()
