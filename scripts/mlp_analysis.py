"""
scripts/mlp_analysis.py
=======================
Section 8 "MLP layer analysis" of the paper.

Extends the Laplace-vs-Gaussian fitting protocol from attention
projections to the MLP sub-layer (feed-forward / up-proj / down-proj
weights).  Reports whether MLP layers show the same architecture-
dependent distributional regime as attention layers.

Architecture-specific token matching:
  GPT-2     → c_fc, c_proj
  BERT      → intermediate.dense, output.dense, fc_in, fc_out
  T5        → wi, wo, wo, wi_0, wi_1, wo
  BART      → fc1, fc2
  RoBERTa   → intermediate.dense, output.dense
  ALBERT   → full_layer_1, full_layer_2

Usage
-----
    python scripts/mlp_analysis.py
"""
from __future__ import annotations

import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.stats import laplace, norm

from ela.analysis import MODEL_IDS

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Tokens that identify MLP / FFN weight matrices
MLP_TOKENS = [
    "mlp", "fc", "ffn", "intermediate", "feed_forward",
    "c_fc", "c_proj", "wi", "wo", "dense",
    "full_layer_1", "full_layer_2",
]
_LAYER_PAT = r"(?:^|\.)(h|layer|layers|block|albert_layers)\.(\d+)"
_LAYER_RE  = __import__("re").compile(_LAYER_PAT)
_EPS = 1e-10


def collect_mlp_tensors(model, max_layers=8):
    groups = {}
    for name, param in model.named_parameters():
        if not name.endswith(".weight") or param.ndim < 2:
            continue
        low = name.lower()
        if not any(t in low for t in MLP_TOKENS):
            continue
        # Exclude anything that still looks like attention
        if any(t in low for t in ("attention", "attn", "query", "key", "value", "q_proj", "k_proj", "v_proj")):
            continue
        m = _LAYER_RE.search(low)
        if not m:
            continue
        groups.setdefault(int(m.group(2)), []).append(
            param.detach().cpu().numpy().reshape(-1)
        )
    tensors = []
    for idx in sorted(groups)[:max_layers]:
        tensors.append(np.concatenate(groups[idx]).astype(np.float64))
    return tensors


def fit_layer(flat: np.ndarray) -> dict:
    loc_l, scale_l = laplace.fit(flat)
    loc_n, scale_n = norm.fit(flat)
    ll_l = float(np.sum(np.log(laplace.pdf(flat, loc_l, scale_l) + _EPS)))
    ll_n = float(np.sum(np.log(norm.pdf(flat,  loc_n, scale_n) + _EPS)))
    return {
        "ll_laplace":  ll_l,
        "ll_gaussian": ll_n,
        "better_fit":  "Laplace" if ll_l > ll_n else "Gaussian",
    }


def analyze_model_mlp(model_id: str, max_layers=8) -> dict | None:
    from transformers import AutoConfig, AutoModel
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    model  = AutoModel.from_pretrained(model_id, trust_remote_code=False)
    tensors = collect_mlp_tensors(model, max_layers=max_layers)
    if not tensors:
        return None
    layers = []
    for i, t in enumerate(tensors):
        r = fit_layer(t)
        layers.append({"layer": i, **r})
    laplace_wins = sum(1 for ln in layers if ln["better_fit"] == "Laplace")
    return {
        "model_id":     model_id,
        "architecture": config.model_type,
        "num_layers":   len(layers),
        "laplace_wins": laplace_wins,
        "gaussian_wins": len(layers) - laplace_wins,
        "laplace_pct":  100.0 * laplace_wins / max(1, len(layers)),
        "layers":       layers,
    }


def main() -> None:
    log.info("=" * 80)
    log.info("MLP LAYER ANALYSIS  (ela-backed)")
    log.info("=" * 80)

    results, skipped, failures = [], [], []
    for mid in MODEL_IDS:
        try:
            log.info("[%s]", mid)
            r = analyze_model_mlp(mid)
            if r is None:
                skipped.append(mid)
                log.info("  skipped: no MLP weights detected")
                continue
            results.append(r)
            log.info("  %d layers  Laplace %d/%d (%.1f%%)",
                     r["num_layers"], r["laplace_wins"], r["num_layers"],
                     r["laplace_pct"])
        except Exception as exc:
            failures.append({"model_id": mid, "error": str(exc)})
            log.info("  FAILED: %s", exc)

    out_path = os.path.join(os.path.dirname(__file__), "..", "results",
                            "mlp_analysis_results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"results": results, "skipped": skipped, "failures": failures},
                  fh, indent=2)
    log.info("\nSaved → %s", out_path)
    log.info("Analyzed: %d  Skipped: %d  Failed: %d",
             len(results), len(skipped), len(failures))


if __name__ == "__main__":
    main()
