#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Config
# -----------------------------
SEED=42
N=50
JUDGE_MODEL="gpt-4o-mini"
WORKERS=2
MAX_TOKENS=512

# -----------------------------
# Files (이미 네가 쓰던 경로로 풀 세팅)
# -----------------------------
# stage1
A_STAGE1="outputs/base_mistral7b/eval_fixed_stage1_200_clean_v4.jsonl"
B_STAGE1="outputs/exp_factual_only/stage1_factual/eval_fixed_stage1_200_clean_v4.jsonl"
A_STAGE1_LABEL="base_stage1"
B_STAGE1_LABEL="stage1_finetune"

# stage2
A_STAGE2="outputs/base_mistral7b/eval_fixed_stage2_200_clean_v4.jsonl"
B_STAGE2="outputs/m2_counsel_only/stage2_counseling/eval_fixed_stage2_200_det_clean_v4.jsonl"
A_STAGE2_LABEL="base_stage2"
B_STAGE2_LABEL="stage2_finetune"

# stage3 base
A_STAGE3="outputs/base_mistral7b/eval_fixed_stage3_200_clean_v4.jsonl"
A_STAGE3_LABEL="base_stage3"

# stage3 warmstart
B_STAGE3_WARM="outputs/exp_stage3_full_warmstart/stage3_full/eval_fixed_stage3_200_clean_v4.jsonl"
B_STAGE3_WARM_LABEL="stage3_warmstart"

# stage3 full(scratch)
B_STAGE3_FULL="outputs/exp_stage3_full_scratch/stage3_full/eval_fixed_stage3_200_clean_v4.jsonl"
B_STAGE3_FULL_LABEL="stage3_full"

# -----------------------------
# Sanity check
# -----------------------------
for f in \
  tools/pairwise_judge.py \
  tools/make_keylist.py \
  tools/subset_by_keylist.py \
  tools/ragas_eval_50.py \
  "$A_STAGE1" "$B_STAGE1" \
  "$A_STAGE2" "$B_STAGE2" \
  "$A_STAGE3" "$B_STAGE3_WARM" "$B_STAGE3_FULL"
do
  if [[ ! -f "$f" ]]; then
    echo "[ERR] missing file: $f"
    exit 1
  fi
done

python -m py_compile \
  tools/pairwise_judge.py \
  tools/make_keylist.py \
  tools/subset_by_keylist.py \
  tools/ragas_eval_50.py

mkdir -p reports/keys /tmp/eval50
mkdir -p reports/pairwise_llm_ref reports/ragas_50

run_one() {
  local NAME="$1"
  local A_PATH="$2"
  local B_PATH="$3"
  local A_LABEL="$4"
  local B_LABEL="$5"

  echo
  echo "============================================================"
  echo "[RUN] $NAME"
  echo "  A=$A_LABEL <- $A_PATH"
  echo "  B=$B_LABEL <- $B_PATH"
  echo "============================================================"

  local KEYFILE="reports/keys/${NAME}_n${N}_seed${SEED}.txt"
  local A_SUB="/tmp/eval50/${NAME}_A_${N}.jsonl"
  local B_SUB="/tmp/eval50/${NAME}_B_${N}.jsonl"

  local PAIR_DIR="reports/pairwise_llm_ref/${NAME}"
  local PAIR_OUT="${PAIR_DIR}/pairwise_${N}_faithful_cleanv4.csv"

  local RAGAS_DIR="reports/ragas_50/${NAME}_proxyref"

  mkdir -p "$PAIR_DIR" "$RAGAS_DIR"

  # 1) key 50 고정
  python tools/make_keylist.py \
    --a "$A_PATH" --b "$B_PATH" \
    --out "$KEYFILE" --n "$N" --seed "$SEED"

  # 2) subset 생성 (A/B 동일 key로)
  python tools/subset_by_keylist.py --in "$A_PATH" --keys "$KEYFILE" --out "$A_SUB"
  python tools/subset_by_keylist.py --in "$B_PATH" --keys "$KEYFILE" --out "$B_SUB"

  # 3) LLM pairwise 50 (faithful + clean_v4)
  python tools/pairwise_judge.py \
    --a "$A_SUB" --b "$B_SUB" \
    --a_label "$A_LABEL" --b_label "$B_LABEL" \
    --out "$PAIR_OUT" \
    --model "$JUDGE_MODEL" --temperature 0 --max_tokens "$MAX_TOKENS" \
    --max "$N" --seed "$SEED" --workers "$WORKERS" \
    --rubric faithful --judge_clean_v4

  # 4) RAGAS proxy 50 (reference를 context로 사용)
  #    -> base/ft 각각 별도 실행해서 평균 비교
  python tools/ragas_eval_50.py \
    --in "$A_SUB" --outdir "$RAGAS_DIR" --label "$A_LABEL" --use_reference_as_context
  python tools/ragas_eval_50.py \
    --in "$B_SUB" --outdir "$RAGAS_DIR" --label "$B_LABEL" --use_reference_as_context

  echo "[DONE] $NAME"
  echo "  Pairwise : $PAIR_OUT"
  echo "  RAGAS    : $RAGAS_DIR/ragas_${A_LABEL}.summary.json , ragas_${B_LABEL}.summary.json"
}

# stage1
run_one "stage1" "$A_STAGE1" "$B_STAGE1" "$A_STAGE1_LABEL" "$B_STAGE1_LABEL"

# stage2
run_one "stage2" "$A_STAGE2" "$B_STAGE2" "$A_STAGE2_LABEL" "$B_STAGE2_LABEL"

# stage3 warmstart vs base
run_one "stage3_warmstart" "$A_STAGE3" "$B_STAGE3_WARM" "$A_STAGE3_LABEL" "$B_STAGE3_WARM_LABEL"

# stage3 full vs base
run_one "stage3_full" "$A_STAGE3" "$B_STAGE3_FULL" "$A_STAGE3_LABEL" "$B_STAGE3_FULL_LABEL"

echo
echo "✅ ALL DONE"
