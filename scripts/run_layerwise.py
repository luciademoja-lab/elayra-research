"""
scripts/run_layerwise.py
========================
Replacement for layerwise_model_comparison.py.

Analyses up to MAX_LAYERS_DEPTH=15 layers per model to reveal the
depth-gradient signature described in Section 4.2 of the paper.

Usage
-----
    python scripts/run_layerwise.py
"""
from __future__ import annotations

import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ela.analysis import MODEL_IDS, MAX_LAYERS_DEPTH, analyze_model

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    log.info("=" * 80)
    log.info("LAYERWISE MODEL COMPARISON  (ela-backed, depth=%d)", MAX_LAYERS_DEPTH)
    log.info("=" * 80)

    all_results = []
    failures = []

    for mid in MODEL_IDS:
        try:
            log.info("[%s] analysing first %d attention layers…", mid, MAX_LAYERS_DEPTH)
            r = analyze_model(mid, max_layers=MAX_LAYERS_DEPTH)
            all_results.append(r)
            log.info("  pretrained: %d layers  %.1f%% Laplace",
                     r["pretrained"]["num_layers"],
                     r["pretrained"]["laplace_pct"])
            log.info("  random-init: %.1f%% Laplace",
                     r["random_init"]["laplace_pct"])
        except Exception as exc:
            failures.append({"model_id": mid, "error": str(exc)})
            log.info("  FAILED: %s", exc)

    out_path = os.path.join(os.path.dirname(__file__), "..", "results",
                            "layerwise_model_comparison.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"results": all_results, "failures": failures}, fh, indent=2)

    log.info("\nSaved %d entries → %s", len(all_results), out_path)
    if all_results:
        log.info("\nSummary (by Laplace%% descending):")
        for item in sorted(all_results,
                           key=lambda x: x["pretrained"]["laplace_pct"],
                           reverse=True):
            p = item["pretrained"]
            log.info("  %-35s  %5.1f%%  (%d/%d layers)",
                     item["model_id"],
                     p["laplace_pct"], p["laplace_wins"], p["num_layers"])


if __name__ == "__main__":
    main()
