#!/usr/bin/env python3
"""Compute token-level KL localization for selected diffs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Token-level KL localization.")
    parser.add_argument("--top_diffs", required=True)
    parser.add_argument("--models_config", required=True)
    parser.add_argument("--run_config", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def maybe_chat_prompt(tokenizer, model_id: str, prompt: str) -> str:
    lower = model_id.lower()
    use_chat = "instruct" in lower or lower.endswith("-it") or "chat" in lower
    if use_chat and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt
    return prompt


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
        "base_model_id": pair["base_model"],
        "instruct_model_id": pair["instruct_model"],
    }


def compute_kl_series(
    prompt: str,
    instruct_model,
    instruct_tok,
    base_model,
    base_tok,
    instruct_model_id: str,
    max_new_tokens: int,
) -> Tuple[List[Dict[str, Any]], str]:
    prompt_text = maybe_chat_prompt(instruct_tok, instruct_model_id, prompt)
    enc = instruct_tok(prompt_text, return_tensors="pt")
    enc = {k: v.to(instruct_model.device) for k, v in enc.items()}
    prompt_len = int(enc["input_ids"].shape[1])

    with torch.no_grad():
        gen = instruct_model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=instruct_tok.pad_token_id,
            eos_token_id=instruct_tok.eos_token_id,
            use_cache=True,
        )

    full_ids = gen[0]
    gen_ids = full_ids[prompt_len:]
    completion_text = instruct_tok.decode(gen_ids, skip_special_tokens=True)

    # Evaluate both models on the same prefix trajectory.
    base_ids = base_tok(prompt_text + completion_text, return_tensors="pt")["input_ids"].to(base_model.device)
    inst_ids = instruct_tok(prompt_text + completion_text, return_tensors="pt")["input_ids"].to(instruct_model.device)

    with torch.no_grad():
        inst_logits = instruct_model(inst_ids).logits[0]
        base_logits = base_model(base_ids).logits[0]

    usable = min(inst_logits.shape[0], base_logits.shape[0])
    prompt_len_shared = min(prompt_len, usable - 1)

    kl_rows: List[Dict[str, Any]] = []
    for step, pos in enumerate(range(prompt_len_shared - 1, usable - 1)):
        p = torch.softmax(inst_logits[pos].float(), dim=-1)
        q = torch.softmax(base_logits[pos].float(), dim=-1)
        q = torch.clamp(q, min=1e-12)
        p = torch.clamp(p, min=1e-12)
        kl = torch.sum(p * (torch.log(p) - torch.log(q))).item()

        token_id = int(inst_ids[0, pos + 1].item())
        token_str = instruct_tok.decode([token_id], skip_special_tokens=False)
        kl_rows.append(
            {
                "step": step,
                "position": int(pos),
                "token_id": token_id,
                "token_str": token_str,
                "kl": float(kl),
            }
        )

    return kl_rows, completion_text


def main() -> None:
    args = parse_args()
    models_cfg = load_yaml(args.models_config)
    run_cfg = load_yaml(args.run_config)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parents[1] / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    paths_cfg = run_cfg.get("paths", {})
    repo_root = Path(__file__).resolve().parents[1]
    kl_dir = out_dir
    fig_dir = repo_root / paths_cfg.get("figures_dir", "results/figures")
    kl_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    top_diffs = json.loads(Path(args.top_diffs).read_text(encoding="utf-8"))
    if not isinstance(top_diffs, list):
        raise ValueError("top_diffs must be a JSON list.")

    if args.smoke:
        top_diffs = top_diffs[:1]

    max_new_tokens = int(run_cfg.get("max_new_tokens", 256))

    pair_map = {p["pair_id"]: p for p in models_cfg.get("pairs", [])}
    runtime_cfg = models_cfg.get("runtime", {})

    loaded_pair_id = None
    loaded = None

    for item in top_diffs:
        diff_id = item.get("diff_id")
        pair_id = item.get("pair_id")
        prompt = item.get("prompt_text", "")
        if not diff_id or not pair_id or not prompt:
            print(f"[WARN] skipping malformed top diff: {item}")
            continue

        if pair_id not in pair_map:
            print(f"[WARN] pair_id not found in config: {pair_id}")
            continue

        if loaded_pair_id != pair_id:
            # Free previous models if any.
            if loaded is not None:
                del loaded
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            print(f"[LOAD] {pair_id}")
            loaded = load_pair_models(pair_map[pair_id], runtime_cfg)
            loaded_pair_id = pair_id

        assert loaded is not None

        kl_rows, completion_text = compute_kl_series(
            prompt=prompt,
            instruct_model=loaded["instruct_model"],
            instruct_tok=loaded["instruct_tok"],
            base_model=loaded["base_model"],
            base_tok=loaded["base_tok"],
            instruct_model_id=loaded["instruct_model_id"],
            max_new_tokens=max_new_tokens if not args.smoke else min(64, max_new_tokens),
        )

        if not kl_rows:
            print(f"[WARN] no KL rows for {diff_id}")
            continue

        top_spikes = sorted(kl_rows, key=lambda r: r["kl"], reverse=True)[:5]
        payload = {
            "diff_id": diff_id,
            "pair_id": pair_id,
            "prompt_id": item.get("prompt_id"),
            "family": item.get("family"),
            "prompt_text": prompt,
            "completion_text": completion_text,
            "kl_by_step": kl_rows,
            "top_spikes": top_spikes,
        }

        kl_path = kl_dir / f"{diff_id}_kl.json"
        kl_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        xs = [r["step"] for r in kl_rows]
        ys = [r["kl"] for r in kl_rows]
        plt.figure(figsize=(10, 4))
        plt.plot(xs, ys, linewidth=1.5)
        for spike in top_spikes:
            plt.axvline(spike["step"], linestyle="--", alpha=0.25)
        plt.title(f"Token-level KL: {diff_id}")
        plt.xlabel("Generation step")
        plt.ylabel("KL(P_instruct || P_base)")
        plt.tight_layout()

        fig_path = fig_dir / f"{diff_id}_kl_plot.png"
        plt.savefig(fig_path, dpi=160)
        plt.close()
        print(f"[OK] wrote {kl_path} and {fig_path}")


if __name__ == "__main__":
    main()
