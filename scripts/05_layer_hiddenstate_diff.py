#!/usr/bin/env python3
"""Compute layerwise hidden-state diffs for selected diffs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer hidden-state diff heatmaps.")
    parser.add_argument("--top_diffs", required=True)
    parser.add_argument("--models_config", required=True)
    parser.add_argument("--run_config", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def parse_dtype(name: str) -> torch.dtype:
    table = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in table:
        raise ValueError(f"Unsupported dtype: {name}")
    return table[name]


def load_pair_models(pair: Dict[str, Any], runtime_cfg: Dict[str, Any]):
    dtype = parse_dtype(runtime_cfg.get("dtype", "float16"))
    device_map = runtime_cfg.get("device_map", "auto")
    load_in_4bit = bool(runtime_cfg.get("load_in_4bit", True))
    trust_remote_code = bool(runtime_cfg.get("trust_remote_code", False))

    def load_one(model_id: str):
        kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                )
            except Exception as exc:
                print(f"[WARN] 4bit unavailable for {model_id}: {exc}")

        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        return model, tok

    base_model, base_tok = load_one(pair["base_model"])
    instruct_model, instruct_tok = load_one(pair["instruct_model"])

    return {
        "base_model": base_model,
        "base_tok": base_tok,
        "instruct_model": instruct_model,
        "instruct_tok": instruct_tok,
    }


def compute_layer_distances(text: str, base_model, base_tok, instruct_model, instruct_tok):
    base_ids = base_tok(text, return_tensors="pt")["input_ids"].to(base_model.device)
    inst_ids = instruct_tok(text, return_tensors="pt")["input_ids"].to(instruct_model.device)

    with torch.no_grad():
        base_out = base_model(base_ids, output_hidden_states=True)
        inst_out = instruct_model(inst_ids, output_hidden_states=True)

    base_h = base_out.hidden_states
    inst_h = inst_out.hidden_states

    n_layers = min(len(base_h), len(inst_h))
    seq_len = min(base_h[0].shape[1], inst_h[0].shape[1])

    cosine_matrix = np.zeros((n_layers, seq_len), dtype=np.float32)
    l2_matrix = np.zeros((n_layers, seq_len), dtype=np.float32)
    mean_l2 = []
    mean_cos = []

    for layer in range(n_layers):
        b = base_h[layer][0, :seq_len, :].float()
        i = inst_h[layer][0, :seq_len, :].float()

        l2 = torch.norm(i - b, dim=-1)
        cos = 1.0 - torch.nn.functional.cosine_similarity(i, b, dim=-1, eps=1e-8)

        l2_np = l2.detach().cpu().numpy()
        cos_np = cos.detach().cpu().numpy()

        l2_matrix[layer] = l2_np
        cosine_matrix[layer] = cos_np
        mean_l2.append(float(l2_np.mean()))
        mean_cos.append(float(cos_np.mean()))

    return {
        "cosine_matrix": cosine_matrix,
        "l2_matrix": l2_matrix,
        "mean_l2": mean_l2,
        "mean_cos": mean_cos,
        "seq_len": int(seq_len),
        "n_layers": int(n_layers),
    }


def main() -> None:
    args = parse_args()
    models_cfg = load_yaml(args.models_config)
    run_cfg = load_yaml(args.run_config)
    repo_root = Path(__file__).resolve().parents[1]

    top_diffs = json.loads(Path(args.top_diffs).read_text(encoding="utf-8"))
    if not isinstance(top_diffs, list):
        raise ValueError("top_diffs must be JSON list")
    if args.smoke:
        top_diffs = top_diffs[:1]

    pair_map = {p["pair_id"]: p for p in models_cfg.get("pairs", [])}
    runtime_cfg = models_cfg.get("runtime", {})

    paths_cfg = run_cfg.get("paths", {})
    kl_dir = repo_root / paths_cfg.get("kl_dir", "results/kl")
    fig_dir = Path(args.out_dir)
    if not fig_dir.is_absolute():
        fig_dir = repo_root / fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    loaded_pair_id = None
    loaded = None

    for diff in top_diffs:
        diff_id = diff.get("diff_id")
        pair_id = diff.get("pair_id")
        if not diff_id or not pair_id:
            continue

        if pair_id not in pair_map:
            print(f"[WARN] Unknown pair_id in top diff: {pair_id}")
            continue

        kl_path = kl_dir / f"{diff_id}_kl.json"
        if not kl_path.exists():
            print(f"[WARN] Missing KL file for {diff_id}: {kl_path}")
            continue

        kl_payload = json.loads(kl_path.read_text(encoding="utf-8"))
        prompt_text = kl_payload.get("prompt_text", "")
        completion_text = kl_payload.get("completion_text", "")
        spike_positions = [int(x.get("position", -1)) for x in kl_payload.get("top_spikes", []) if isinstance(x, dict)]
        text = f"{prompt_text}{completion_text}"

        if loaded_pair_id != pair_id:
            if loaded is not None:
                del loaded
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print(f"[LOAD] {pair_id}")
            loaded = load_pair_models(pair_map[pair_id], runtime_cfg)
            loaded_pair_id = pair_id

        assert loaded is not None
        dist = compute_layer_distances(
            text=text,
            base_model=loaded["base_model"],
            base_tok=loaded["base_tok"],
            instruct_model=loaded["instruct_model"],
            instruct_tok=loaded["instruct_tok"],
        )

        cos_mat = dist["cosine_matrix"]
        mean_l2 = dist["mean_l2"]

        # Optional spike-only summary.
        valid_spikes = [p for p in spike_positions if 0 <= p < cos_mat.shape[1]]
        spike_l2_mean = []
        if valid_spikes:
            for layer in range(cos_mat.shape[0]):
                spike_vals = dist["l2_matrix"][layer, valid_spikes]
                spike_l2_mean.append(float(np.mean(spike_vals)))

        plt.figure(figsize=(12, 6))
        ax1 = plt.subplot(1, 2, 1)
        im = ax1.imshow(cos_mat, aspect="auto", origin="lower", interpolation="nearest")
        ax1.set_title(f"Layer x Token Cosine Distance\n{diff_id}")
        ax1.set_xlabel("Token position")
        ax1.set_ylabel("Layer")
        for p in valid_spikes:
            ax1.axvline(p, color="white", linestyle="--", alpha=0.3)
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(mean_l2, label="mean L2 / layer")
        if spike_l2_mean:
            ax2.plot(spike_l2_mean, label="spike-token mean L2 / layer")
        ax2.set_title("Layerwise mean distance")
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Distance")
        ax2.legend()
        plt.tight_layout()

        fig_path = fig_dir / f"{diff_id}_layer_heatmap.png"
        plt.savefig(fig_path, dpi=160)
        plt.close()

        stats_path = fig_dir / f"{diff_id}_layer_stats.json"
        stats = {
            "diff_id": diff_id,
            "pair_id": pair_id,
            "seq_len": dist["seq_len"],
            "n_layers": dist["n_layers"],
            "mean_l2": mean_l2,
            "mean_cos": dist["mean_cos"],
            "spike_positions": valid_spikes,
            "spike_l2_mean": spike_l2_mean,
        }
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"[OK] wrote {fig_path}")


if __name__ == "__main__":
    main()
