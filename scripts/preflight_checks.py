#!/usr/bin/env python3
"""Preflight checks for Colab execution readiness."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

import torch

from common import load_yaml


def bytes_to_gb(n: int) -> float:
    return round(n / (1024 ** 3), 2)


def check_hf_access(models: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"logged_in": False, "models": {}}
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        who = api.whoami()
        out["logged_in"] = True
        out["whoami"] = who.get("name") if isinstance(who, dict) else str(who)

        for m in models:
            try:
                info = api.model_info(m)
                out["models"][m] = {
                    "ok": True,
                    "gated": bool(getattr(info, "gated", False)),
                }
            except Exception as exc:
                out["models"][m] = {"ok": False, "error": str(exc)}
    except Exception as exc:
        out["error"] = str(exc)
    return out


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    models_cfg = load_yaml(repo_root / "configs" / "models.yaml")

    disk = shutil.disk_usage(repo_root)
    model_ids = []
    for p in models_cfg.get("pairs", []):
        model_ids.append(p.get("base_model", ""))
        model_ids.append(p.get("instruct_model", ""))
    model_ids = [m for m in model_ids if m]

    hf = check_hf_access(model_ids)
    gpu_ready = bool(torch.cuda.is_available())
    gemini_key = bool(os.environ.get("GEMINI_API_KEY"))

    report = {
        "cwd": str(repo_root),
        "python": os.environ.get("PYTHON_VERSION", "unknown"),
        "torch_version": torch.__version__,
        "cuda_available": gpu_ready,
        "cuda_device_count": torch.cuda.device_count() if gpu_ready else 0,
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        "gemini_key_present": gemini_key,
        "disk_total_gb": bytes_to_gb(disk.total),
        "disk_used_gb": bytes_to_gb(disk.used),
        "disk_free_gb": bytes_to_gb(disk.free),
        "hf": hf,
        "model_ids": model_ids,
    }

    hard_failures = []
    if not gpu_ready:
        hard_failures.append("CUDA GPU not available (Colab GPU runtime required for planned full run).")
    if not gemini_key:
        hard_failures.append("GEMINI_API_KEY missing (required for llm judge + ask-LLM baseline).")
    if report["disk_free_gb"] < 20:
        hard_failures.append("Low disk free space (<20GB); may fail model downloads/cache.")

    # If HF is logged in, require model access checks to pass.
    if hf.get("logged_in"):
        for model_id, info in hf.get("models", {}).items():
            if not info.get("ok"):
                hard_failures.append(f"HF access check failed for {model_id}: {info.get('error')}")
    else:
        hard_failures.append("Hugging Face not authenticated (required for Gemma access).")

    report["hard_failures"] = hard_failures
    report["ready_for_full_run"] = len(hard_failures) == 0

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
