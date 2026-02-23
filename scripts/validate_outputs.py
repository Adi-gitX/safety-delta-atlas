#!/usr/bin/env python3
"""Validate prompt and result schemas for Safety Delta Atlas."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List

PROMPT_FIELDS = {"id", "family", "prompt", "notes"}
PARA_FIELDS = {"id", "paraphrase_1", "paraphrase_2"}
RAW_FIELDS = {
    "model_id",
    "model_role",
    "pair_id",
    "prompt_id",
    "family",
    "variant",
    "temp",
    "seed",
    "prompt_text",
    "output_text",
}
JUDGED_EXTRA_FIELDS = {
    "judge_type",
    "label_refusal",
    "label_honesty",
    "label_sycophancy",
    "judge_rationale",
}


def iter_jsonl(path: str):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object line in {path}")
            yield obj


def require_fields(row: Dict, required: set, ctx: str) -> None:
    missing = required - set(row.keys())
    if missing:
        raise ValueError(f"{ctx}: missing fields {sorted(missing)}")


def check_jsonl_files(paths: List[str], required: set, ctx: str) -> int:
    count = 0
    for path in paths:
        for i, row in enumerate(iter_jsonl(path), start=1):
            require_fields(row, required, f"{ctx} {path}:{i}")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate outputs.")
    parser.add_argument("--repo_root", default=str(Path(__file__).resolve().parents[1]))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.repo_root)

    prompt_files = glob.glob(str(root / "data/prompts/*.jsonl"))
    prompt_regular = [p for p in prompt_files if not p.endswith("_paraphrases.jsonl")]
    prompt_para = [p for p in prompt_files if p.endswith("_paraphrases.jsonl")]

    n_prompt = check_jsonl_files(prompt_regular, PROMPT_FIELDS, "prompt")
    n_para = check_jsonl_files(prompt_para, PARA_FIELDS, "paraphrase")

    raw_files = glob.glob(str(root / "results/raw_generations/*.jsonl"))
    judged_files = glob.glob(str(root / "results/judged/*.jsonl"))

    n_raw = check_jsonl_files(raw_files, RAW_FIELDS, "raw") if raw_files else 0
    n_judged = (
        check_jsonl_files(judged_files, RAW_FIELDS.union(JUDGED_EXTRA_FIELDS), "judged")
        if judged_files
        else 0
    )

    print(f"[OK] prompts validated: {n_prompt} rows")
    print(f"[OK] paraphrases validated: {n_para} rows")
    print(f"[OK] raw validated: {n_raw} rows")
    print(f"[OK] judged validated: {n_judged} rows")


if __name__ == "__main__":
    main()
