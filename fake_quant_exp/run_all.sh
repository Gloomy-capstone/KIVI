#!/usr/bin/env bash
# KIVI 논문 Table 1 / Table 3 fake quantization 6개 configuration을
# Llama-2-7B로 모두 실행한다.
#
# 결과는 results/*.json 으로 저장되고, 마지막에 summary 표가 출력된다.

set -euo pipefail

# 가상환경 활성화
source /mnt/data/gloomyteam/kivi/bin/activate

# ===== 사용자 환경에 맞게 수정 =====
MODEL="${MODEL:-meta-llama/Llama-2-7b-hf}"   # 로컬 경로면 그 경로로 export MODEL=...
TASKS="${TASKS:-coqa,truthfulqa_gen}"
GROUP_SIZE="${GROUP_SIZE:-32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LIMIT_ARG=""                                  # 빠른 sanity check면 LIMIT_ARG="--limit 50"
# ===================================

OUT_DIR="${OUT_DIR:-results}"
mkdir -p "$OUT_DIR"

# (이름, k_bits, v_bits, k_dim, v_dim) 6개 — 논문 Table 1 / Table 3과 동일
CONFIGS=(
  "16bit          16  16  token    token"
  "4bit_KT_VT     4   4   token    token"
  "2bit_KT_VT     2   2   token    token"
  "2bit_KC_VC     2   2   channel  channel"
  "2bit_KT_VC     2   2   token    channel"
  "2bit_KC_VT     2   2   channel  token"
)

for cfg in "${CONFIGS[@]}"; do
  read -r NAME KB VB KD VD <<< "$cfg"
  echo
  echo "=================================================="
  echo "  Running: $NAME  (K=${KB}bit/${KD}, V=${VB}bit/${VD})"
  echo "=================================================="

  python run_eval.py \
    --model "$MODEL" \
    --tasks "$TASKS" \
    --k_bits "$KB" --v_bits "$VB" \
    --k_dim "$KD" --v_dim "$VD" \
    --group_size "$GROUP_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --output "$OUT_DIR/${NAME}.json" \
    $LIMIT_ARG
done

echo
echo "=================================================="
echo "  All runs finished. Summarising..."
echo "=================================================="
python summarize.py "$OUT_DIR"
