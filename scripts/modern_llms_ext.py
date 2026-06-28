"""
scripts/modern_llms_ext.py
==========================
Section 8 "Extension to modern large language models" of the paper.

Tests whether the Laplace-dominant pattern generalises to contemporary
architectures (LLaMA-3.2-1B is included; larger models are excluded
because they exceed the 4 GB VRAM of the development GPU — see README §
"Hardware limits").

Usage
-----
    python scripts/modern_llms_ext.py           # safe models only
    python scripts/modern_llms_ext.py --all     # also attempt >1 B models
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ela.analysis import (
    MAX_LAYERS_PRIMARY,
    MAX_LAYERS_DEPTH,
    analyze_model,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Models that safely fit 4 GB VRAM (fp16 on GTX 1060).
SAFE_MODELS = [
    "meta-llama/Llama-3.2-1B",  # ~2 GB fp16
]

# Additional candidates (>1 B parameters, require >4 GB or CPU-only fallback).
LARGE_MODELS = [
    # "microsoft/phi-2",        # ~2.7 B params, ~5.4 GB fp16 — borderline
    # "EleutherAI/gpt-j-6b",   # 6 B params — well over 4 GB
]

ALL_MODELS = SAFE_MODELS + LARGE_MODELS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="Also attempt large-models (may OOM)")
    ap.add_argument("--depth", action="store_true",
                    help="Use MAX_LAYERS_DEPTH (15) instead of 8")
    args = ap.parse_args()

    max_layers = MAX_LAYERS_DEPTH if args.depth else MAX_LAYERS_PRIMARY
    model_ids = ALL_MODELS if args.all else SAFE_MODELS

    log.info("=" * 80)
    log.info("MODERN LLM EXTENSION  (ela-backed, max_layers=%d)", max_layers)
    log.info("Models: %s", model_ids)
    log.info("=" * 80)

    results, failures = [], []
    for mid in model_ids:
        try:
            log.info("[%s]…", mid)
            r = analyze_model(mid, max_layers=max_layers)
            results.append(r)
            log.info("  pretrained: %.1f%% Laplace  (%d/%d layers)",
                     r["pretrained"]["laplace_pct"],
                     r["pretrained"]["laplace_wins"],
                     r["pretrained"]["num_layers"])
            log.info("  random-init: %.1f%% Laplace",
                     r["random_init"]["laplace_pct"])
        except Exception as exc:
            failures.append({"model_id": mid, "error": str(exc)})
            log.info("  FAILED (skipped): %s", exc)

    out = {"results": results, "failures": failures, "model_count": len(results)}
    out_path = os.path.join(os.path.dirname(__file__), "..", "results",
                            "modern_llm_results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    log.info("\nSaved → %s  (%d successful, %d failures)",
             out_path, len(results), len(failures))

    if results:
        log.info("\nTop Laplace prevalence:")
        for item in sorted(results,
                           key=lambda x: x["pretrained"]["laplace_pct"],
                           reverse=True)[:5]:
            log.info("  %-35s  %.1f%%", item["model_id"],
                     item["pretrained"]["laplace_pct"])


if __name__ == "__main__":
    main()
