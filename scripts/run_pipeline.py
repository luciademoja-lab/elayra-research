"""
scripts/run_pipeline.py
=======================
Replacement for broader_analysis_pipeline.py.

Runs the 15-model (primary) analysis: for each model, fits both pretrained
and randomly-initialised variants across MAX_LAYERS_PRIMARY layers.

Bootstrap 95 % CI is computed and included in the output JSON.

Usage
-----
    python scripts/run_pipeline.py                  # 15 primary models
    INCLUDE_OPTIONAL_MODELS=1 python scripts/run_pipeline.py   # + Llama-1B
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from tqdm import tqdm

from ela.analysis import (
    CAPABLE_MODEL_IDS,
    MODEL_IDS,
    MAX_LAYERS_PRIMARY,
    analyze_model,
)
from ela.bootstrap import bootstrap_ci
from ela.config import PipelineConfig
from ela.utils import flush_cuda

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    cfg = PipelineConfig()
    include_opt = os.environ.get("INCLUDE_OPTIONAL_MODELS", "0") == "1"
    model_ids = CAPABLE_MODEL_IDS if include_opt else MODEL_IDS

    log.info("=" * 80)
    log.info("BROADER ANALYSIS PIPELINE  (ela-backed)")
    log.info("=" * 80)
    log.info("Models: %d  (%s optional)", len(model_ids),
             "with" if include_opt else "no")

    results: List[Dict] = []
    failures: List[Dict] = []

    for mid in tqdm(model_ids, desc="Models", unit="model"):
        try:
            log.info("[%s]", mid)
            r = analyze_model(mid, max_layers=cfg.max_layers)
            results.append(r)
            log.info("  pretrained Laplace%%: %.1f", r["pretrained"]["laplace_pct"])
            log.info("  random-init Laplace%%: %.1f",
                     r["random_init"]["laplace_pct"])
            log.info("  init kurtosis: %.4f", r["init_stats"]["kurtosis"])
        except Exception as exc:
            failures.append({"model_id": mid, "error": str(exc)})
            log.info("  FAILED: %s", exc)

    # Deterministic GPU cache release after model loop
    flush_cuda()

    # ------------------------------------------------------------------ #
    # Bootstrap CI for Spearman ρ: init_kurtosis  vs  pretrained laplace% #
    # ------------------------------------------------------------------ #
    if len(results) >= 3:
        kurt_vals  = [r["init_stats"]["kurtosis"] for r in results]
        lap_pcts   = [r["pretrained"]["laplace_pct"]  for r in results]
        ci = bootstrap_ci(kurt_vals, lap_pcts, n=cfg.bootstrap_n, seed=cfg.bootstrap_seed)
        boot = {
            "spearman_rho":  round(ci["stat"], 4),
            "ci_95_low":     round(ci["ci_low"], 4),
            "ci_95_high":    round(ci["ci_high"], 4),
            "n_bootstrap":   int(ci["n"]),
        }
        log.info("\nBootstrap Spearman ρ = %.4f  CI95 [%.4f, %.4f]",
                 boot["spearman_rho"], boot["ci_95_low"], boot["ci_95_high"])
    else:
        boot = {}

    out = {
        "results":     results,
        "failures":    failures,
        "model_count": len(results),
        "bootstrap":   boot,
        "meta": {
            "max_layers": cfg.max_layers,
            "include_optional": include_opt,
        },
    }
    out_path = os.path.join(os.path.dirname(__file__), "..", "results",
                            "broader_analysis_results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    log.info("\nSaved %d results → %s", len(results), out_path)


if __name__ == "__main__":
    main()
