"""
scripts/head_level_analysis.py
===============================
Section 8 "head-level analysis within layers" of the paper.

Fits Laplace vs Gaussian to *individual attention heads* (not the full
layer concatenation) to reveal within-layer heterogeneity.

For architectures with separate q_proj/k_proj/v_proj projections (BERT,
RoBERTa, etc.) the head dimension is inferred from the weight shape and
num_attention_heads config.  For GPT-2's fused c_attn the head dim is
now decomposed natively (see ela.analysis.collect_head_tensors).

Outputs
-------
results/head_level_results.json  — per-head statistics
results/head_level_heatmap.png   — model×layer binary heat-map

Usage
-----
    python scripts/head_level_analysis.py
"""
from __future__ import annotations

import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.stats import laplace, norm
from tqdm import tqdm

from ela.analysis import MODEL_IDS, collect_head_tensors, collect_attention_tensors
from ela.utils import flush_cuda
from ela.viz import layer_heatmap

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

_MAX_LAYERS = 12
_EPS = 1e-10


def _head_winner(head_vec: np.ndarray) -> str:
    flat = head_vec.astype(np.float64)
    loc_l, scale_l = laplace.fit(flat)
    loc_n, scale_n = norm.fit(flat)
    ll_l = float(np.sum(np.log(laplace.pdf(flat, loc_l, scale_l) + _EPS)))
    ll_n = float(np.sum(np.log(norm.pdf(flat,  loc_n, scale_n) + _EPS)))
    return "Laplace" if ll_l > ll_n else "Gaussian"


def analyze_model_headwise(model_id: str) -> dict:
    from transformers import AutoConfig, AutoModel
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    model  = AutoModel.from_pretrained(model_id, trust_remote_code=False)

    head_groups = collect_head_tensors(model)
    if not head_groups:
        # Fallback: treat the whole layer as a single "head"
        log.info("  [%s] no per-head access — falling back to whole-layer", model_id)
        tensors = collect_attention_tensors(model, max_layers=_MAX_LAYERS)
        layers = []
        for idx, t in enumerate(tensors):
            layers.append({
                "layer": idx,
                "n_heads": 1,
                "laplace_heads": 1 if _head_winner(t) == "Laplace" else 0,
                "gaussian_heads": 1 if _head_winner(t) == "Gaussian" else 0,
                "laplace_pct": 100.0 if _head_winner(t) == "Laplace" else 0.0,
                "better_fit": _head_winner(t),
                "head_details": [{"head": 0, "fit": _head_winner(t)}],
            })
        del model
        flush_cuda()
        return {"model_id": model_id, "architecture": config.model_type,
                "layers": layers, "note": "fallback (whole-layer as single head)"}

    layers = []
    for layer_idx in sorted(head_groups)[:_MAX_LAYERS]:
        head_vecs = head_groups[layer_idx]
        n = len(head_vecs)
        wins = sum(1 for v in head_vecs if _head_winner(v) == "Laplace")
        details = [{"head": h, "fit": _head_winner(v)} for h, v in enumerate(head_vecs)]
        layers.append({
            "layer":       layer_idx,
            "n_heads":     n,
            "laplace_heads": wins,
            "gaussian_heads": n - wins,
            "laplace_pct": 100.0 * wins / max(1, n),
            "better_fit":  "Laplace" if wins > n / 2 else "Gaussian",
            "head_details": details,
        })
    del model
    flush_cuda()
    return {"model_id": model_id, "architecture": config.model_type, "layers": layers}


def main() -> None:
    log.info("=" * 80)
    log.info("HEAD-LEVEL ANALYSIS  (ela-backed)")
    log.info("=" * 80)

    all_results = []
    failures = []
    for mid in tqdm(MODEL_IDS, desc="Models", unit="model"):
        try:
            log.info("[%s]", mid)
            r = analyze_model_headwise(mid)
            all_results.append(r)
            for ln in r["layers"]:
                log.info("  layer %2d: %5.1f%% Laplace  (%d/%d heads)",
                         ln["layer"], ln["laplace_pct"],
                         ln["laplace_heads"], ln["n_heads"])
        except Exception as exc:
            failures.append({"model_id": mid, "error": str(exc)})
            log.info("  FAILED: %s", exc)

    out_path = os.path.join(os.path.dirname(__file__), "..", "results",
                            "head_level_results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"results": all_results, "failures": failures}, fh, indent=2)
    log.info("\nSaved JSON → %s", out_path)

    # Re-use the layer heatmap from ela.viz
    png_path = layer_heatmap(
        all_results,
        title="Head-level Laplace Dominance (per layer)",
        out_path=os.path.join(os.path.dirname(__file__), "..", "results",
                               "head_level_heatmap.png"),
    )
    log.info("Saved PNG → %s", png_path)


if __name__ == "__main__":
    main()
