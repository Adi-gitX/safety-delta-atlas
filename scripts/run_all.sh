#!/usr/bin/env bash
set -euo pipefail

MODE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$MODE" ]]; then
  echo "Usage: bash scripts/run_all.sh --mode {preflight|prompts|generate|judge|score|kl|layer|cards|validate|full|smoke}"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_CFG="$ROOT_DIR/configs/models.yaml"
RUN_CFG="$ROOT_DIR/configs/run.yaml"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Python not found in PATH."
  exit 1
fi

RAW_DIR="$ROOT_DIR/results/raw_generations"
JUDGED_DIR="$ROOT_DIR/results/judged"
SCORED_DIR="$ROOT_DIR/results/scored"
FIG_DIR="$ROOT_DIR/results/figures"
CARDS_DIR="$ROOT_DIR/results/diff_cards"
TOP_DIFFS="$SCORED_DIR/top_diffs.json"
SUMMARY_CSV="$SCORED_DIR/summary_tables.csv"

mkdir -p "$RAW_DIR" "$JUDGED_DIR" "$SCORED_DIR" "$FIG_DIR" "$CARDS_DIR" "$ROOT_DIR/results/kl"

run_prompts() {
  "$PYTHON_BIN" "$ROOT_DIR/scripts/00_make_prompt_datasets.py"
}

run_preflight() {
  "$PYTHON_BIN" "$ROOT_DIR/scripts/preflight_checks.py"
}

run_generate() {
  local smoke_flag="${1:-}"
  export RUN_CFG_PATH="$RUN_CFG"
  PAIR_A_FAMILIES=$("$PYTHON_BIN" - <<'PY'
import os
import yaml
cfg=yaml.safe_load(open(os.environ['RUN_CFG_PATH']))
print(','.join(cfg.get('pair_a_families', [])))
PY
)
  PAIR_B_FAMILIES=$("$PYTHON_BIN" - <<'PY'
import os
import yaml
cfg=yaml.safe_load(open(os.environ['RUN_CFG_PATH']))
print(','.join(cfg.get('pair_b_families', [])))
PY
)

  "$PYTHON_BIN" "$ROOT_DIR/scripts/01_generate_outputs.py" \
    --models_config "$MODELS_CFG" \
    --run_config "$RUN_CFG" \
    --families "$PAIR_A_FAMILIES" \
    --pair_scope pair_a \
    --out_dir "$RAW_DIR" \
    $smoke_flag

  "$PYTHON_BIN" "$ROOT_DIR/scripts/01_generate_outputs.py" \
    --models_config "$MODELS_CFG" \
    --run_config "$RUN_CFG" \
    --families "$PAIR_B_FAMILIES" \
    --pair_scope pair_b \
    --out_dir "$RAW_DIR" \
    $smoke_flag
}

run_judge() {
  local smoke_flag="${1:-}"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/02_judge_outputs.py" \
    --in_glob "$RAW_DIR/*.jsonl" \
    --run_config "$RUN_CFG" \
    --judge_types "regex,llm,random" \
    --out_dir "$JUDGED_DIR" \
    $smoke_flag
}

run_score() {
  local smoke_flag="${1:-}"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/03_score_stability_generalization.py" \
    --judged_glob "$JUDGED_DIR/*.jsonl" \
    --models_config "$MODELS_CFG" \
    --run_config "$RUN_CFG" \
    --out_dir "$SCORED_DIR" \
    $smoke_flag
}

run_kl() {
  local smoke_flag="${1:-}"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/04_token_kl_localization.py" \
    --top_diffs "$TOP_DIFFS" \
    --models_config "$MODELS_CFG" \
    --run_config "$RUN_CFG" \
    --out_dir "$ROOT_DIR/results/kl" \
    $smoke_flag
}

run_layer() {
  local smoke_flag="${1:-}"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/05_layer_hiddenstate_diff.py" \
    --top_diffs "$TOP_DIFFS" \
    --models_config "$MODELS_CFG" \
    --run_config "$RUN_CFG" \
    --out_dir "$FIG_DIR" \
    $smoke_flag
}

run_cards() {
  "$PYTHON_BIN" "$ROOT_DIR/scripts/06_make_diff_cards.py" \
    --top_diffs "$TOP_DIFFS" \
    --summary_csv "$SUMMARY_CSV" \
    --fig_dir "$FIG_DIR" \
    --out_dir "$CARDS_DIR"
}

run_validate() {
  "$PYTHON_BIN" "$ROOT_DIR/scripts/validate_outputs.py" --repo_root "$ROOT_DIR"
}

case "$MODE" in
  preflight)
    run_preflight
    ;;
  prompts)
    run_prompts
    ;;
  generate)
    run_generate
    ;;
  judge)
    run_judge
    ;;
  score)
    run_score
    ;;
  kl)
    run_kl
    ;;
  layer)
    run_layer
    ;;
  cards)
    run_cards
    ;;
  validate)
    run_validate
    ;;
  full)
    run_prompts
    run_generate
    run_judge
    run_score
    run_kl
    run_layer
    run_cards
    run_validate
    ;;
  smoke)
    run_prompts
    run_generate "--smoke"
    run_judge "--smoke"
    run_score "--smoke"
    run_kl "--smoke"
    run_layer "--smoke"
    run_cards
    run_validate
    ;;
  *)
    echo "Unknown mode: $MODE"
    exit 1
    ;;
esac

echo "[DONE] mode=$MODE"
