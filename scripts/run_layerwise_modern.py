"""
scripts/run_layerwise_modern.py
===============================
Add modern decoder-only LLMs — LLaMA-3.2-1B and Phi-2 — to the layerwise
4-way distribution analysis, and MERGE their entries into the existing
`results/layerwise_model_comparison.json` (same schema as the 15-model run).

Why a separate script
---------------------
These two models were left out of the first run for two DIFFERENT reasons,
neither of which is a property of the distribution analysis itself:

  * LLaMA-3.2-1B — gated HuggingFace repo (403). Needs an approved access
    request + a token; see RUNBOOK. Not a code problem.
  * Phi-2 — excluded by the 4 GB VRAM budget. But the distribution fits run on
    CPU (scipy); loading a model to read its weights does NOT need the GPU. The
    real constraint here is system RAM, not VRAM. Phi-2 (2.7 B) fits on a
    typical workstation's RAM.

This script is memory-lean: it loads one model at a time and never holds the
pretrained and random-init copies simultaneously, so a 2.7 B model stays
tractable. It relies on the `.float()`-before-`.numpy()` fix in
`ela.analysis.collect_attention_tensors` (modern checkpoints are bf16, which
numpy cannot convert directly).

Usage
-----
    # LLaMA needs a token + an approved access form first (see RUNBOOK):
    #   $env:HF_TOKEN = "hf_..."     (PowerShell)
    python scripts/run_layerwise_modern.py                 # both models
    python scripts/run_layerwise_modern.py --models microsoft/phi-2
    python scripts/run_layerwise_modern.py --no-random     # skip random-init (saves RAM/time)

Output
------
    Merged into results/layerwise_model_comparison.json (replacing any entry
    with the same model_id; existing 15 entries are preserved).
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.getLogger("ela.analysis").setLevel(logging.WARNING)
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

from ela.analysis import (
    MAX_LAYERS_DEPTH,
    collect_attention_tensors,
    summarize_initialization,
    summarize_layerwise_fit,
)
from ela.utils import flush_cuda

log = logging.getLogger("modern")
logging.basicConfig(level=logging.INFO, format="%(message)s")

MODERN_MODEL_IDS = [
    "meta-llama/Llama-3.2-1B",   # gated: needs approved HF access + HF_TOKEN
    "microsoft/phi-2",           # CPU-analysable despite the 4 GB VRAM budget
]

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "layerwise_model_comparison.json"
)


def _gated_hint(model_id: str, exc: Exception) -> None:
    msg = str(exc).lower()
    if any(t in msg for t in ("gated", "401", "403", "restricted", "awaiting a review", "access to model")):
        log.info("  ACCESS BLOCKED for %s. This is a HuggingFace gating issue, not a bug.", model_id)
        log.info("  Fix: (1) log in at huggingface.co with the token's account;")
        log.info("       (2) open https://huggingface.co/%s and click 'Agree and access';", model_id)
        log.info("       (3) set the token in this shell: $env:HF_TOKEN = \"hf_...\"; then rerun.")
    elif "trust_remote_code" in msg or "unrecognized" in msg or "unknown" in msg:
        log.info("  %s may need a newer transformers. Try: uv pip install -U transformers", model_id)


def analyze_modern(model_id: str, max_layers: int, do_random: bool) -> dict:
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    entry: dict = {"model_id": model_id, "architecture": getattr(config, "model_type", "unknown")}

    # --- pretrained (native dtype, low CPU memory; freed before random-init) ---
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=False, torch_dtype="auto", low_cpu_mem_usage=True
    )
    try:
        tensors = collect_attention_tensors(model, max_layers=max_layers)
        entry["pretrained"] = summarize_layerwise_fit(tensors, include_student_t=True)
    finally:
        del model
        gc.collect()
        flush_cuda()

    # --- random-init (optional; doubles peak memory, so it is opt-outable) ---
    if do_random:
        rand = AutoModel.from_config(config)
        try:
            entry["random_init"] = summarize_layerwise_fit(
                collect_attention_tensors(rand, max_layers=max_layers), include_student_t=True
            )
            entry["init_stats"] = summarize_initialization(rand)
        finally:
            del rand
            gc.collect()
            flush_cuda()

    return entry


def _merge_into_checkpoint(entries: list, failures: list) -> None:
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        data = {"results": [], "failures": []}
    data.setdefault("results", [])
    data.setdefault("failures", [])

    by_id = {r["model_id"]: i for i, r in enumerate(data["results"])}
    for e in entries:
        if e["model_id"] in by_id:
            data["results"][by_id[e["model_id"]]] = e   # replace stale entry
        else:
            data["results"].append(e)
    # drop any previous failure records for models we just succeeded on
    done_ids = {e["model_id"] for e in entries}
    data["failures"] = [f for f in data.get("failures", []) if f.get("model_id") not in done_ids]
    data["failures"].extend(failures)

    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def _median_beta(summary: dict):
    betas = sorted(l["ggd_beta"] for l in summary["layers"] if "ggd_beta" in l)
    return betas[len(betas) // 2] if betas else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None, help="Default: LLaMA-3.2-1B + Phi-2.")
    ap.add_argument("--max-layers", type=int, default=MAX_LAYERS_DEPTH)
    ap.add_argument("--no-random", action="store_true", help="Skip random-init pass (saves RAM/time).")
    args = ap.parse_args()

    model_ids = args.models if args.models else MODERN_MODEL_IDS
    entries, failures = [], []
    for mid in model_ids:
        log.info("[%s] fitting up to %d layers ...", mid, args.max_layers)
        try:
            entries.append(analyze_modern(mid, args.max_layers, do_random=not args.no_random))
            log.info("  ok")
        except Exception as exc:  # noqa: BLE001
            failures.append({"model_id": mid, "error": str(exc)})
            log.info("  FAILED: %s", exc)
            _gated_hint(mid, exc)

    _merge_into_checkpoint(entries, failures)

    print("\n" + "=" * 72)
    print(f"{'model':26}{'layers':>8}{'2way Lap%':>11}{'median beta':>14}")
    print("-" * 72)
    for e in entries:
        p = e["pretrained"]
        print(f"{e['model_id'][:26]:26}{p['num_layers']:>8}{p['laplace_pct']:>10.1f}%{_median_beta(p):>14.3f}")
    print("-" * 72)
    print(f"Merged into {os.path.relpath(CHECKPOINT_PATH)}  (added/updated {len(entries)}, failed {len(failures)})")
    if failures:
        print("Failed: " + ", ".join(f["model_id"] for f in failures) + "  — see messages above.")


if __name__ == "__main__":
    main()
