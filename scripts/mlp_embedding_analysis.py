"""
scripts/mlp_embedding_analysis.py
=================================
Extends the 4-way distribution fit (Laplace / Gaussian / Student-t / GGD,
BIC-selected — the same `summarize_layerwise_fit` used for attention) from
attention projections to the two weight groups that actually dominate a
transformer's memory footprint: the MLP / feed-forward sub-layers and the
embedding matrices.

Why this exists (see STATE_OF_EVIDENCE.md §3): the paper's sub-Laplacian
finding is currently characterized for *attention* weights only. Attention is a
minority of the bytes; MLP and embeddings are the bulk. Any whole-model
compression / on-device claim needs the same GGD-beta characterization here.
This script produces it, plus a byte-share census so the beta values can be
weighted by the fraction of the model they actually cover.

Design notes
------------
- Reuses `ela.analysis.summarize_layerwise_fit(..., include_student_t=True)`
  verbatim, so MLP/embedding results are schema-identical to the attention file
  (`results/layerwise_model_comparison.json`) and reconcilable the same way.
- Deterministic subsampling is inherited from `ela` (ELA_SUBSAMPLE / seed), so
  fits are comparable to the attention analysis and reproducible on any machine.
- Output: full detail to JSON; console prints ONLY a compact final table.
  (No per-layer console spam — run it, read one table.)

Usage
-----
    python scripts/mlp_embedding_analysis.py                  # all 15 models
    python scripts/mlp_embedding_analysis.py --models t5-small   # quick single check
    python scripts/mlp_embedding_analysis.py --max-layers 15  # match attention depth

Output
------
    results/mlp_embedding_4way.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

# Keep third-party libraries quiet — the whole point is a clean console.
logging.getLogger("ela.analysis").setLevel(logging.WARNING)
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

from ela.analysis import (
    MODEL_IDS,
    ATTENTION_TOKENS,
    _LAYER_PATTERN,
    summarize_layerwise_fit,
)
from ela.utils import flush_cuda

log = logging.getLogger("mlp_emb")
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Tokens identifying MLP / feed-forward weight matrices (mirrors mlp_analysis.py).
MLP_TOKENS = [
    "mlp", "fc", "ffn", "intermediate", "feed_forward",
    "c_fc", "c_proj", "wi", "wo", "dense",
    "full_layer_1", "full_layer_2",
]

# Tokens identifying embedding matrices across the 15-model zoo.
EMB_TOKENS = [
    "wte", "wpe",                       # GPT-2 (word / position)
    "word_embeddings", "position_embeddings", "token_type_embeddings",  # BERT/RoBERTa/ELECTRA/ALBERT
    "embed_tokens", "embed_positions",  # BART / T5 encoder-decoder
    "shared",                           # T5 / mT5 tied embedding
]

# Names that look like embeddings but are NOT dense embedding matrices to fit.
EMB_EXCLUDE = ["layernorm", "layer_norm", "relative_attention_bias", "ln_"]


def _is_attention(name_l: str) -> bool:
    return any(t in name_l for t in ATTENTION_TOKENS)


def _is_mlp(name_l: str) -> bool:
    return any(t in name_l for t in MLP_TOKENS) and not _is_attention(name_l)


def _is_embedding(name_l: str) -> bool:
    if any(x in name_l for x in EMB_EXCLUDE):
        return False
    return any(t in name_l for t in EMB_TOKENS)


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------

def collect_mlp_tensors(model, max_layers: int):
    """One flattened float64 array per MLP layer index (like the attention collector)."""
    groups = {}
    for name, param in model.named_parameters():
        if not name.endswith(".weight") or param.ndim < 2:
            continue
        low = name.lower()
        if not _is_mlp(low):
            continue
        m = _LAYER_PATTERN.search(low)
        if not m:
            continue
        groups.setdefault(int(m.group(2)), []).append(
            param.detach().cpu().float().numpy().reshape(-1)  # .float(): bf16-safe
        )
    tensors = []
    for idx in sorted(groups)[:max_layers]:
        tensors.append(np.concatenate(groups[idx]).astype(np.float64))
    return tensors


def collect_embedding_tensors(model):
    """One flattened float64 array per DISTINCT embedding matrix.

    Deduplicates tied tensors (e.g. T5 `shared` == encoder/decoder `embed_tokens`)
    by underlying storage pointer, so a tied embedding is fitted once.
    """
    seen_ptrs = set()
    tensors, labels = [], []
    for name, param in model.named_parameters():
        if param.ndim < 2:
            continue
        low = name.lower()
        if not _is_embedding(low):
            continue
        ptr = param.data_ptr()
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)
        tensors.append(param.detach().cpu().float().numpy().reshape(-1).astype(np.float64))  # .float(): bf16-safe
        labels.append(name)
    return tensors, labels


def param_census(model):
    """Unique-parameter counts by category (dedup tied weights by storage pointer)."""
    seen = set()
    counts = {"embedding": 0, "attention": 0, "mlp": 0, "other": 0}
    for name, param in model.named_parameters():
        ptr = param.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        n = int(param.numel())
        low = name.lower()
        if _is_embedding(low):
            counts["embedding"] += n
        elif _is_attention(low):
            counts["attention"] += n
        elif _is_mlp(low):
            counts["mlp"] += n
        else:
            counts["other"] += n
    counts["total"] = sum(v for k, v in counts.items() if k != "total")
    return counts


def _median_beta(summary):
    betas = sorted(l["ggd_beta"] for l in summary["layers"] if "ggd_beta" in l)
    return betas[len(betas) // 2] if betas else None


def _bic_counts(summary):
    out = {}
    for l in summary["layers"]:
        w = l.get("bic_winner")
        out[w] = out.get(w, 0) + 1
    return out


# ---------------------------------------------------------------------------
# Per-model driver
# ---------------------------------------------------------------------------

def analyze_one(model_id: str, max_layers: int) -> dict:
    from transformers import AutoConfig, AutoModel
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=False)
    try:
        census = param_census(model)
        mlp_tensors = collect_mlp_tensors(model, max_layers=max_layers)
        emb_tensors, emb_labels = collect_embedding_tensors(model)
        mlp_summary = summarize_layerwise_fit(mlp_tensors, include_student_t=True) if mlp_tensors else None
        emb_summary = summarize_layerwise_fit(emb_tensors, include_student_t=True) if emb_tensors else None
    finally:
        del model
        import gc
        gc.collect()
        flush_cuda()
    if emb_summary is not None:
        for entry, lbl in zip(emb_summary["layers"], emb_labels):
            entry["name"] = lbl
    return {
        "model_id": model_id,
        "architecture": config.model_type,
        "param_census": census,
        "mlp": mlp_summary,
        "embedding": emb_summary,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="Subset of model ids (default: all 15).")
    ap.add_argument("--max-layers", type=int, default=15,
                    help="Max MLP layers to fit per model (default 15).")
    ap.add_argument("--out", default=None, help="Output JSON path.")
    args = ap.parse_args()

    model_ids = args.models if args.models else MODEL_IDS
    out_path = args.out or os.path.join(
        os.path.dirname(__file__), "..", "results", "mlp_embedding_4way.json"
    )

    results, failures = [], []
    for mid in model_ids:
        try:
            log.info("fitting %s ...", mid)
            results.append(analyze_one(mid, max_layers=args.max_layers))
        except Exception as exc:  # noqa: BLE001
            failures.append({"model_id": mid, "error": str(exc)})
            log.info("  FAILED: %s", exc)

    payload = {
        "results": results,
        "failures": failures,
        "meta": {
            "max_layers": args.max_layers,
            "note": "4-way (Laplace/Gaussian/Student-t/GGD) BIC fit on MLP and "
                    "embedding weights; param_census dedupes tied tensors.",
        },
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    # -------- compact console summary (the only thing printed) --------
    print("\n" + "=" * 78)
    print(f"{'model':22}{'emb%':>7}{'attn%':>7}{'mlp%':>7}{'β mlp':>9}{'β emb':>9}  {'mlp BIC winners'}")
    print("-" * 78)
    for r in results:
        c = r["param_census"]
        tot = max(1, c["total"])
        emb_p = 100 * c["embedding"] / tot
        att_p = 100 * c["attention"] / tot
        mlp_p = 100 * c["mlp"] / tot
        bmlp = _median_beta(r["mlp"]) if r["mlp"] else None
        bemb = _median_beta(r["embedding"]) if r["embedding"] else None
        bic = _bic_counts(r["mlp"]) if r["mlp"] else {}
        bmlp_s = f"{bmlp:.3f}" if bmlp is not None else "  n/a"
        bemb_s = f"{bemb:.3f}" if bemb is not None else "  n/a"
        print(f"{r['model_id'][:22]:22}{emb_p:6.1f}%{att_p:6.1f}%{mlp_p:6.1f}%"
              f"{bmlp_s:>9}{bemb_s:>9}  {bic}")
    print("-" * 78)
    print(f"Saved full detail -> {os.path.relpath(out_path)}")
    print(f"Models fitted: {len(results)}   Failed: {len(failures)}")
    print("Reading key: beta<1 sharply peaked (very compressible), beta~2 Gaussian.")
    print("The compression lever's real size = (mlp% + emb%) weighted by how low their beta is.")


if __name__ == "__main__":
    main()
