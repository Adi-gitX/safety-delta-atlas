#!/usr/bin/env python3
"""Generate Base vs Instruct outputs across prompt families."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import append_jsonl, iter_jsonl, load_yaml, model_id_to_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate outputs for model pairs.")
    parser.add_argument("--models_config", required=True)
    parser.add_argument("--run_config", required=True)
    parser.add_argument("--families", default="all", help="Comma list or 'all'.")
    parser.add_argument(
        "--pair_scope",
        default="all",
        help="all|pair_a|pair_b|comma-separated pair_ids",
    )
    parser.add_argument("--prompt_dir", default="")
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


def should_use_chat_template(model_id: str) -> bool:
    lower = model_id.lower()
    return "instruct" in lower or lower.endswith("-it") or "chat" in lower


def load_prompt_bundle(prompt_dir: Path, family: str) -> List[Dict[str, Any]]:
    base_path = prompt_dir / f"{family}.jsonl"
    para_path = prompt_dir / f"{family}_paraphrases.jsonl"

    if not base_path.exists():
        raise FileNotFoundError(f"Missing prompt file: {base_path}")
    if not para_path.exists():
        raise FileNotFoundError(f"Missing paraphrase file: {para_path}")

    base_rows = list(iter_jsonl(base_path))
    para_rows = list(iter_jsonl(para_path))
    para_by_id = {row["id"]: row for row in para_rows}

    all_rows: List[Dict[str, Any]] = []
    for row in base_rows:
        prompt_id = row.get("id")
        if not prompt_id:
            raise ValueError(f"Prompt row missing id in {base_path}")
        if row.get("family") != family:
            raise ValueError(f"Family mismatch for {prompt_id}: {row.get('family')} != {family}")

        para = para_by_id.get(prompt_id)
        if not para:
            raise ValueError(f"Missing paraphrases for prompt id {prompt_id}")

        all_rows.append(
            {
                "prompt_id": prompt_id,
                "family": family,
                "variant": "orig",
                "prompt_text": row.get("prompt", ""),
                "notes": row.get("notes", ""),
            }
        )
        all_rows.append(
            {
                "prompt_id": prompt_id,
                "family": family,
                "variant": "paraphrase_1",
                "prompt_text": para.get("paraphrase_1", ""),
                "notes": row.get("notes", ""),
            }
        )
        all_rows.append(
            {
                "prompt_id": prompt_id,
                "family": family,
                "variant": "paraphrase_2",
                "prompt_text": para.get("paraphrase_2", ""),
                "notes": row.get("notes", ""),
            }
        )

    return all_rows


def load_model_and_tokenizer(
    model_id: str,
    dtype_name: str,
    device_map: str,
    load_in_4bit: bool,
    trust_remote_code: bool,
):
    dtype = parse_dtype(dtype_name)
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
            print(f"[WARN] 4bit quantization unavailable for {model_id}: {exc}")

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def prompt_to_model_input(tokenizer, prompt_text: str, use_chat: bool) -> str:
    if use_chat and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt_text
    return prompt_text


def deterministic_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_existing_keys(out_path: Path) -> set[Tuple[str, str, str, float, int]]:
    keys: set[Tuple[str, str, str, float, int]] = set()
    if not out_path.exists():
        return keys
    for row in iter_jsonl(out_path):
        key = (
            row.get("pair_id", ""),
            row.get("prompt_id", ""),
            row.get("variant", ""),
            float(row.get("temp", 0.0)),
            int(row.get("seed", 0)),
        )
        keys.add(key)
    return keys


def select_pairs(models_cfg: Dict[str, Any], scope: str) -> List[Dict[str, Any]]:
    pairs = models_cfg.get("pairs", [])
    if scope == "all":
        return list(pairs)
    if scope == "pair_a":
        return [pairs[0]]
    if scope == "pair_b":
        return [pairs[1]]
    allow = {x.strip() for x in scope.split(",") if x.strip()}
    chosen = [p for p in pairs if p.get("pair_id") in allow]
    if not chosen:
        raise ValueError(f"No pairs match scope: {scope}")
    return chosen


def select_families(run_cfg: Dict[str, Any], families_arg: str) -> List[str]:
    if families_arg != "all":
        return [x.strip() for x in families_arg.split(",") if x.strip()]
    fams = []
    fams.extend(run_cfg.get("pair_a_families", []))
    fams.extend(run_cfg.get("pair_b_families", []))
    seen = set()
    out = []
    for f in fams:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def maybe_smoke_filter(
    prompt_rows: List[Dict[str, Any]],
    seeds: List[int],
    temps: List[float],
    smoke: bool,
) -> tuple[List[Dict[str, Any]], List[int], List[float]]:
    if not smoke:
        return prompt_rows, seeds, temps

    prompt_ids = sorted({row["prompt_id"] for row in prompt_rows})[:6]
    keep = set(prompt_ids)
    prompt_rows = [row for row in prompt_rows if row["prompt_id"] in keep]
    return prompt_rows, seeds[:1], temps[:1]


def main() -> None:
    args = parse_args()

    models_cfg = load_yaml(args.models_config)
    run_cfg = load_yaml(args.run_config)

    run_paths = run_cfg.get("paths", {})
    repo_root = Path(__file__).resolve().parents[1]
    prompt_dir = Path(args.prompt_dir or run_paths.get("prompt_dir", "data/prompts"))
    if not prompt_dir.is_absolute():
        prompt_dir = repo_root / prompt_dir

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_families = select_families(run_cfg, args.families)
    pair_list = select_pairs(models_cfg, args.pair_scope)

    seeds = [int(x) for x in run_cfg.get("seeds", [11, 22, 33])]
    temps = [float(x) for x in run_cfg.get("temperatures", [0.2, 0.8])]
    max_new_tokens = int(run_cfg.get("max_new_tokens", 256))

    runtime_cfg = models_cfg.get("runtime", {})
    dtype_name = runtime_cfg.get("dtype", "float16")
    device_map = str(runtime_cfg.get("device_map", "auto"))
    load_in_4bit = bool(runtime_cfg.get("load_in_4bit", True))
    trust_remote_code = bool(runtime_cfg.get("trust_remote_code", False))

    prompt_rows: List[Dict[str, Any]] = []
    for family in all_families:
        prompt_rows.extend(load_prompt_bundle(prompt_dir, family))

    prompt_rows, seeds, temps = maybe_smoke_filter(prompt_rows, seeds, temps, args.smoke)
    print(
        f"Loaded {len(prompt_rows)} prompt variants across {len(all_families)} families; "
        f"seeds={seeds}, temps={temps}, smoke={args.smoke}"
    )

    for pair in pair_list:
        pair_id = pair["pair_id"]
        for role_key, model_key in (("base", "base_model"), ("instruct", "instruct_model")):
            model_id = pair[model_key]
            print(f"\n[MODEL] Loading {model_id} ({role_key}) for pair {pair_id}")
            model, tokenizer = load_model_and_tokenizer(
                model_id=model_id,
                dtype_name=dtype_name,
                device_map=device_map,
                load_in_4bit=load_in_4bit,
                trust_remote_code=trust_remote_code,
            )

            out_path = out_dir / model_id_to_filename(model_id)
            existing = build_existing_keys(out_path)

            use_chat = should_use_chat_template(model_id)
            total = len(prompt_rows) * len(seeds) * len(temps)
            progress = tqdm(total=total, desc=f"{model_id}")
            write_count = 0

            try:
                for row in prompt_rows:
                    for temp in temps:
                        for seed in seeds:
                            key = (pair_id, row["prompt_id"], row["variant"], temp, seed)
                            if key in existing:
                                progress.update(1)
                                continue

                            deterministic_seed(seed)
                            input_text = prompt_to_model_input(tokenizer, row["prompt_text"], use_chat)
                            encoded = tokenizer(input_text, return_tensors="pt")
                            encoded = {k: v.to(model.device) for k, v in encoded.items()}
                            input_len = encoded["input_ids"].shape[1]

                            do_sample = temp > 0
                            generate_kwargs = {
                                "max_new_tokens": max_new_tokens,
                                "do_sample": do_sample,
                                "pad_token_id": tokenizer.pad_token_id,
                                "eos_token_id": tokenizer.eos_token_id,
                                "use_cache": True,
                            }
                            if do_sample:
                                generate_kwargs["temperature"] = temp

                            t0 = time.time()
                            with torch.no_grad():
                                out = model.generate(**encoded, **generate_kwargs)
                            elapsed = time.time() - t0

                            new_tokens = out[0][input_len:]
                            output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

                            record = {
                                "model_id": model_id,
                                "model_role": role_key,
                                "pair_id": pair_id,
                                "prompt_id": row["prompt_id"],
                                "family": row["family"],
                                "variant": row["variant"],
                                "temp": temp,
                                "seed": seed,
                                "prompt_text": row["prompt_text"],
                                "output_text": output_text,
                                "latency_seconds": round(elapsed, 4),
                            }
                            append_jsonl(out_path, record)
                            write_count += 1
                            existing.add(key)
                            progress.update(1)
            finally:
                progress.close()
                # Free memory before loading next model.
                del model
                del tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            print(f"[MODEL] Wrote {write_count} new rows to {out_path}")


if __name__ == "__main__":
    main()
