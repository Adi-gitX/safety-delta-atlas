#!/usr/bin/env python3
"""Create markdown diff cards for selected top diffs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make diff cards.")
    parser.add_argument("--top_diffs", required=True)
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--fig_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def sanitize_prompt(text: str, max_len: int = 220) -> str:
    cleaned = " ".join((text or "").strip().split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3] + "..."


def fmt_pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def bool_yes_no(v: Any) -> str:
    return "yes" if bool(v) else "no"


def main() -> None:
    args = parse_args()

    top_diffs = json.loads(Path(args.top_diffs).read_text(encoding="utf-8"))
    if not isinstance(top_diffs, list):
        raise ValueError("top_diffs must be JSON list")

    summary = pd.read_csv(args.summary_csv)
    fig_dir = Path(args.fig_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for k, diff in enumerate(top_diffs, start=1):
        diff_id = diff["diff_id"]
        pair_id = diff["pair_id"]
        family = diff["family"]
        prompt_id = diff["prompt_id"]
        prompt_text = sanitize_prompt(diff.get("prompt_text", ""))

        sub = summary[
            (summary["pair_id"] == pair_id)
            & (summary["family"] == family)
            & (summary["prompt_id"] == prompt_id)
        ].copy()

        by_judge = {}
        for judge in ["llm", "regex", "random"]:
            ss = sub[sub["judge_type"] == judge]
            if len(ss) > 0:
                row = ss.iloc[0]
                by_judge[judge] = {
                    "base_rate": float(row["base_rate"]),
                    "instruct_rate": float(row["instruct_rate"]),
                    "diff_magnitude": float(row["diff_magnitude"]),
                    "stability_score": float(row["stability_score"]),
                    "generalizes_to_pair_b": bool(row.get("generalizes_to_pair_b", False)),
                }

        primary = by_judge.get(diff.get("judge_type", "llm"), None)
        if primary is None and by_judge:
            primary = next(iter(by_judge.values()))

        diff_name = f"{family.replace('_', ' ').title()} delta for {prompt_id}"
        kl_fig = fig_dir / f"{diff_id}_kl_plot.png"
        layer_fig = fig_dir / f"{diff_id}_layer_heatmap.png"

        observed = (
            f"Base positive-rate {fmt_pct(diff['base_rate'])} vs Instruct {fmt_pct(diff['instruct_rate'])}; "
            f"absolute delta {fmt_pct(diff['diff_magnitude'])}."
        )

        hypotheses = [
            "Instruction tuning shifts response priors for this behavior family in a consistent direction.",
            "The observed delta is robust to paraphrases and decoding resampling for this prompt.",
            "KL spikes indicate a small number of generation positions where policy-conditioned behavior diverges.",
        ]
        if float(diff.get("resampling_drop", 0.0)) > 0.05:
            hypotheses.append("Some apparent effect size likely came from decoding variance (resampling drop is non-trivial).")

        alternatives = [
            "Prompt-template effects may contribute to part of the measured delta.",
            "Judge-label noise (especially regex/random) may inflate or deflate true behavioral change.",
            "Differences in tokenizer segmentation could shift KL and hidden-state alignment artifacts.",
        ]

        next_experiments = [
            "Re-run with a second prompt subset in the same family to check prompt-id dependence.",
            "Repeat localization at greedy and temperature=0.2 to test decoding dependence.",
            "Add one larger model family to test scale dependence of the same diff.",
        ]

        lines = []
        lines.append(f"# Diff Card {k}: {diff_name}")
        lines.append("")
        lines.append(f"## Diff name\n{diff_name}")
        lines.append("")
        lines.append(f"## Prompt (sanitized)\n`{prompt_text}`")
        lines.append("")
        lines.append(f"## Observed behavior difference\n{observed}")
        lines.append("")
        lines.append("## Stability metrics")
        lines.append(f"- Stability score: {fmt_pct(diff['stability_score'])}")
        lines.append(f"- Resampling drop: {fmt_pct(diff.get('resampling_drop', 0.0))}")
        lines.append("")
        lines.append("## Generalization")
        lines.append(f"- Pair B generalization: {bool_yes_no(diff.get('generalizes_to_pair_b', False))}")
        lines.append(f"- Generalization score: {fmt_pct(diff.get('generalization_score', 0.0))}")
        lines.append("")
        lines.append("## Baselines comparison (regex vs llm-judge vs random)")
        for judge in ["llm", "regex", "random"]:
            if judge in by_judge:
                bj = by_judge[judge]
                lines.append(
                    f"- {judge}: delta {fmt_pct(bj['diff_magnitude'])}, stability {fmt_pct(bj['stability_score'])}, "
                    f"base {fmt_pct(bj['base_rate'])} -> instruct {fmt_pct(bj['instruct_rate'])}"
                )
            else:
                lines.append(f"- {judge}: unavailable")
        lines.append("")
        lines.append("## KL plot + layer heatmap")
        lines.append(f"![KL plot](../figures/{kl_fig.name})")
        lines.append(f"![Layer heatmap](../figures/{layer_fig.name})")
        lines.append("")
        lines.append("## What I think is going on")
        for h in hypotheses[:4]:
            lines.append(f"- {h}")
        lines.append("")
        lines.append("## Alternative explanations")
        for a in alternatives[:4]:
            lines.append(f"- {a}")
        lines.append("")
        lines.append("## Next experiments")
        for n in next_experiments[:4]:
            lines.append(f"- {n}")
        lines.append("")

        out_path = out_dir / f"diff_{k}.md"
        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
