# Safety Delta Atlas

Safety Delta Atlas is a baseline-heavy pipeline for finding stable, safety-relevant behavioral diffs between Base and Instruct models.
It compares modern open models across refusal style, honesty/uncertainty, and sycophancy/policy-compliance prompt families.
It includes random, regex, and Gemini LLM-judge baselines, plus resampling and ask-an-LLM baselines.
It localizes selected diffs with token-level KL spikes and layerwise hidden-state distance heatmaps.
It outputs reproducible artifacts: scored tables, diff cards, figures, and final write-up documents.

Default model pairs:
- Pair A: `google/gemma-2-2b` vs `google/gemma-2-2b-it`
- Pair B: `Qwen/Qwen2.5-3B` vs `Qwen/Qwen2.5-3B-Instruct`

## Reproducibility

### 1) Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 1.1) Auth requirements (Gemma + Gemini)
```bash
hf auth login
```
- Ensure your Hugging Face account has accepted Gemma terms for `google/gemma-2-2b` and `google/gemma-2-2b-it`.
- Export Gemini key before `--mode judge` and `--mode score`:
```bash
export GEMINI_API_KEY=<YOUR_KEY>
```

### 2) Full pipeline
```bash
bash scripts/run_all.sh --mode full
```

### 2.1) Preflight (recommended before full run)
```bash
bash scripts/run_all.sh --mode preflight
```

### 3) Smoke pipeline
```bash
bash scripts/run_all.sh --mode smoke
```

## Results locations
- Raw generations: `results/raw_generations/`
- Judged outputs: `results/judged/`
- Scores and baselines: `results/scored/`
- KL localization JSON: `results/kl/`
- Figures (KL + heatmaps): `results/figures/`
- Diff cards: `results/diff_cards/`

## Colab notes
- Use a GPU runtime (T4/L4 preferred).
- Log in to Hugging Face in Colab (`hf auth login`) for Gemma access.
- Set `GEMINI_API_KEY` in secrets/environment before judge and score steps.
