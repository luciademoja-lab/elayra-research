"""
scripts/init_analysis.py
=======================
Replacement for multi_model_init_analysis.py.

Loads a randomly-initialised version of each primary model, computes
initialisation kurtosis/std/range, and stores results in
``results/expanded_model_init_results.json``.

Usage
-----
    python scripts/init_analysis.py
"""
from __future__ import annotations

import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tqdm import tqdm

import numpy as np
from scipy.stats import kurtosis, skew
from transformers import AutoConfig, AutoModel

from ela.analysis import MODEL_IDS, collect_attention_tensors, MAX_LAYERS_DEPTH
from ela.utils import flush_cuda

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _init_stats(model_id: str) -> dict:
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    model = AutoModel.from_config(config)  # random init
    tensors = collect_attention_tensors(model, max_layers=MAX_LAYERS_DEPTH)
    flat = np.concatenate(tensors).astype(np.float64).reshape(-1)
    result = {
        "model_id":         model_id,
        "architecture":     config.model_type,
        "num_params":       sum(p.numel() for p in model.parameters()),
        "weight_mean":      float(np.mean(flat)),
        "weight_std":       float(np.std(flat)),
        "weight_range":     float(np.max(flat) - np.min(flat)),
        "weight_kurtosis":  float(kurtosis(flat)),
        "weight_skewness":  float(skew(flat)),
    }
    del model
    return result


def main() -> None:
    log.info("=" * 72)
    log.info("EXPANDED INITIALIZATION ANALYSIS  (ela-backed)")
    log.info("=" * 72)

    results, failures = [], []
    for mid in tqdm(MODEL_IDS, desc="Models", unit="model"):
        try:
            log.info("[%s]", mid)
            s = _init_stats(mid)
            results.append(s)
            log.info("  kurtosis=%.4f  std=%.6f  params=%d",
                     s["weight_kurtosis"], s["weight_std"], s["num_params"])
        except Exception as exc:
            failures.append({"model_id": mid, "error": str(exc)})
            log.info("  FAILED: %s", exc)

    flush_cuda()

    out_path = os.path.join(os.path.dirname(__file__), "..", "results",
                            "expanded_model_init_results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"results": results, "failures": failures}, fh, indent=2)

    log.info("\nSaved → %s", out_path)
    if results:
        log.info("\nTop kurtosis:")
        for item in sorted(results, key=lambda x: x["weight_kurtosis"], reverse=True)[:10]:
            log.info("  %-35s  %.4f", item["model_id"], item["weight_kurtosis"])


if __name__ == "__main__":
    main()
