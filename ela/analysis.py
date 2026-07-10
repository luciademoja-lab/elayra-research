"""
ela/analysis.py
===============
Core analysis primitives — model loading, weight collection, layer-wise
distribution fitting, and per-model summaries.

All heavy numpy/torch work is in this single file; other ela/ modules
import from here and are purely procedural wrappers.
"""

from __future__ import annotations

import gc
import logging
import os
import re
from typing import Dict, List

import numpy as np
import torch
from scipy.stats import kurtosis, laplace, norm
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM

from .utils import flush_cuda

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Primary 15-model set (tested throughout the paper).
MODEL_IDS: List[str] = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "bert-base-uncased",
    "bert-large-uncased",
    "distilbert-base-uncased",
    "roberta-base",
    "roberta-large",
    "distilroberta-base",
    "albert-base-v2",
    "google/electra-small-discriminator",
    "facebook/bart-base",
    "t5-small",
    "t5-base",
    "google/mt5-small",
]

# Candidate extensions — NOT all fit 4 GB VRAM.  See README §"Hardware limits".
OPTIONAL_MODEL_IDS: List[str] = [
    "meta-llama/Llama-3.2-1B",      # ~2 GB fp16 — marginal
    # "EleutherAI/gpt-j-6b",         # excluded: >6 GB
    # "microsoft/phi-2",             # excluded: >4 GB
    # "google/vit-base-patch16-224", # excluded: ViT is separate analysis
]

# Full capable set = primary + those optional models that fit 4 GB VRAM.
# GTX 1060 4 GB: Llama-3.2-1B in fp16 needs ~2 GB (close but viable with
# offloading; included here for completeness, user may remove).
CAPABLE_MODEL_IDS: List[str] = MODEL_IDS + [
    m for m in OPTIONAL_MODEL_IDS
    if "Llama-3.2-1B" in m             # only the 1 B model is safe
]

# Max layers analysed per model in the *primary* (8-layer) protocol.
MAX_LAYERS_PRIMARY = 8

# Max layers in the *layer-depth* protocol.
MAX_LAYERS_DEPTH = 15

ATTENTION_TOKENS = [
    "attention", "selfattention", "attn", "q_proj", "k_proj",
    "v_proj", "query", "key", "value", "c_attn", "in_proj_weight",
    "q_lin", "k_lin", "v_lin",
]

# Regex: matches depth indicators like h.3, layer.5, layers.2, block.7,
# albert_layers.0 (the digit after the keyword is the layer index).
_LAYER_PATTERN = re.compile(r"(?:^|\.)(h|layer|layers|block|albert_layers)\.(\d+)")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weight collection
# ---------------------------------------------------------------------------

def collect_attention_tensors(
    model: torch.nn.Module,
    max_layers: int = MAX_LAYERS_PRIMARY,
) -> List[np.ndarray]:
    """Return one flattened float64 numpy array per attention layer.

    Parameters
    ----------
    model:
        A loaded transformers model (pretrained or from-config).
    max_layers:
        Stop after this many layers; caps cross-model comparability.

    Raises
    ------
    ValueError:
        If no attention-weight parameters are detected for the model.
    """
    layer_groups: Dict[int, List[np.ndarray]] = {}

    for name, param in model.named_parameters():
        if not name.endswith(".weight") or param.ndim < 2:
            continue
        lowered = name.lower()
        if not any(tok in lowered for tok in ATTENTION_TOKENS):
            continue
        match = _LAYER_PATTERN.search(lowered)
        if not match:
            continue
        layer_idx = int(match.group(2))
        layer_groups.setdefault(layer_idx, []).append(
            param.detach().cpu().numpy().reshape(-1)
        )

    if not layer_groups:
        raise ValueError(
            f"No attention weights found for model {getattr(model, 'name_or_path', '?')}"
        )

    tensors: List[np.ndarray] = []
    for idx in sorted(layer_groups)[:max_layers]:
        tensors.append(np.concatenate(layer_groups[idx]).astype(np.float64))

    if not tensors:
        raise ValueError("No layerwise attention tensors were assembled")
    return tensors


# ---------------------------------------------------------------------------
# Distribution fitting
# ---------------------------------------------------------------------------

# Layer subsampling for tractable MLE fitting (esp. Student-t, whose
# iterative fit on full tensors of >10^6 elements is the dominant cost).
# ELA_SUBSAMPLE=0 disables subsampling (full-tensor fits, slow).
SUBSAMPLE_N: int = int(os.environ.get("ELA_SUBSAMPLE", "500000"))
SUBSAMPLE_SEED: int = int(os.environ.get("ELA_SUBSAMPLE_SEED", "12345"))


def _maybe_subsample(flat: np.ndarray, layer_idx: int) -> np.ndarray:
    """Deterministically subsample a flattened weight vector.

    Uses a per-layer seed (SUBSAMPLE_SEED + layer_idx) so every run — on any
    machine — fits exactly the same subsample. Returns the input unchanged
    when subsampling is disabled or the tensor is already small enough.
    """
    if SUBSAMPLE_N <= 0 or flat.size <= SUBSAMPLE_N:
        return flat
    rng = np.random.default_rng(SUBSAMPLE_SEED + layer_idx)
    return rng.choice(flat, size=SUBSAMPLE_N, replace=False)


def summarize_layerwise_fit(
    tensors: List[np.ndarray],
    include_student_t: bool = False,
) -> Dict[str, object]:
    """Fit Laplace / Gaussian (optionally Student-t) to each layer tensor.

    Layers larger than ELA_SUBSAMPLE elements (default 500 000) are
    deterministically subsampled before fitting; per-layer provenance is
    recorded in the output (``n_total``, ``n_used``).

    Returns
    -------
    dict with keys: num_layers, laplace_wins, gaussian_wins,
                    laplace_pct, layers (list of per-layer dicts), and
                    optionally student_t_wins / student_t_pct.
    """
    if SUBSAMPLE_N > 0:
        logger.info(
            "Layer fitting with deterministic subsampling: n=%d, seed=%d "
            "(set ELA_SUBSAMPLE=0 for full-tensor fits)",
            SUBSAMPLE_N, SUBSAMPLE_SEED,
        )
    results = []
    for idx, weights in enumerate(tensors):
        full = weights.reshape(-1).astype(np.float64)
        flat = _maybe_subsample(full, idx)
        loc_l, scale_l = laplace.fit(flat)
        loc_n, scale_n = norm.fit(flat)
        ll_laplace  = float(np.sum(np.log(laplace.pdf(flat, loc_l, scale_l) + 1e-10)))
        ll_gaussian = float(np.sum(np.log(norm.pdf(flat, loc_n, scale_n) + 1e-10)))
        entry: Dict[str, object] = {
            "layer": idx,
            "n_total": int(full.size),
            "n_used": int(flat.size),
            "ll_laplace": ll_laplace,
            "ll_gaussian": ll_gaussian,
            "better_fit": "Laplace" if ll_laplace > ll_gaussian else "Gaussian",
        }
        if include_student_t:
            from scipy.stats import gennorm, t as student_t
            try:
                # Initial guesses (df=5, robust loc/scale) cut fit iterations
                # substantially versus scipy's generic starting point.
                df, loc_t, scale_t = student_t.fit(
                    flat, 5.0, loc=float(np.median(flat)), scale=float(np.std(flat)),
                )
                ll_student = float(np.sum(np.log(student_t.pdf(flat, df, loc_t, scale_t) + 1e-10)))
                entry["student_t_df"] = float(df)
            except Exception:
                ll_student = -float("inf")
            entry["ll_student_t"] = ll_student
            # Generalized Gaussian (GGD): shape beta continuously interpolates
            # Laplace (beta=1) and Gaussian (beta=2); exponential-type tails.
            # The 3-param GGD-vs-Student-t duel discriminates exponential vs
            # power-law tail regimes at equal model complexity.
            try:
                beta, loc_g2, scale_g2 = gennorm.fit(
                    flat, 1.5, loc=float(np.median(flat)), scale=float(np.std(flat)),
                )
                ll_ggd = float(np.sum(np.log(gennorm.pdf(flat, beta, loc_g2, scale_g2) + 1e-10)))
                entry["ggd_beta"] = float(beta)
            except Exception:
                ll_ggd = -float("inf")
            entry["ll_ggd"] = ll_ggd
            # Classify by BIC, not raw log-likelihood. Student-t and GGD have a
            # third free parameter and nest/approach the Gaussian (df -> inf,
            # beta = 2), so raw LL is biased toward them: on synthetic Gaussian
            # data the t "wins" with df in the hundreds. AIC/BIC penalise the
            # extra parameter and recover the true generating family. n is
            # large enough here that AIC and BIC almost always agree.
            n = flat.size
            k = {"Laplace": 2, "Gaussian": 2, "Student-t": 3, "GGD": 3}
            ll = {"Laplace": ll_laplace, "Gaussian": ll_gaussian,
                  "Student-t": ll_student, "GGD": ll_ggd}
            aic = {d: 2 * k[d] - 2 * ll[d] for d in ll}
            bic = {d: k[d] * np.log(n) - 2 * ll[d] for d in ll}
            entry["aic_winner"] = min(aic, key=aic.get)
            entry["bic_winner"] = min(bic, key=bic.get)
            entry["rawll_winner"] = max(ll, key=ll.get)
            # Primary classification uses BIC (most conservative on the df penalty).
            entry["better_fit"] = entry["bic_winner"]
        results.append(entry)

    laplace_wins  = sum(1 for r in results if r["better_fit"] == "Laplace")
    gaussian_wins = sum(1 for r in results if r["better_fit"] == "Gaussian")
    num = len(results)
    out: Dict[str, object] = {
        "num_layers": num,
        "laplace_wins": laplace_wins,
        "gaussian_wins": gaussian_wins,
        "laplace_pct": 100.0 * laplace_wins / max(1, num),
        "subsample_n": SUBSAMPLE_N,
        "subsample_seed": SUBSAMPLE_SEED,
        "layers": results,
    }
    if include_student_t:
        st_wins = sum(1 for r in results if r["better_fit"] == "Student-t")
        out["student_t_wins"] = st_wins
        out["student_t_pct"]  = 100.0 * st_wins / max(1, num)
        ggd_wins = sum(1 for r in results if r["better_fit"] == "GGD")
        out["ggd_wins"] = ggd_wins
        out["ggd_pct"]  = 100.0 * ggd_wins / max(1, num)
    return out


# ---------------------------------------------------------------------------
# Initialisation statistics
# ---------------------------------------------------------------------------

def summarize_initialization(model: torch.nn.Module) -> Dict[str, float]:
    """Return mean, std, range, kurtosis, skewness over all attention weights."""
    tensors = collect_attention_tensors(model, max_layers=MAX_LAYERS_DEPTH)
    flat = np.concatenate(tensors).astype(np.float64).reshape(-1)
    return {
        "mean":     float(np.mean(flat)),
        "std":      float(np.std(flat)),
        "range":    float(np.max(flat) - np.min(flat)),
        "kurtosis": float(kurtosis(flat)),
        "skewness": float(np.mean((flat - np.mean(flat)) ** 3) / (np.std(flat) ** 3)),
    }


# ---------------------------------------------------------------------------
# Full per-model analysis
# ---------------------------------------------------------------------------

def analyze_model(
    model_id: str,
    max_layers: int = MAX_LAYERS_PRIMARY,
    include_student_t: bool = False,
) -> Dict[str, object]:
    """Analyse one model at pretrained + random-init.

    Returns
    -------
    dict:
        model_id, architecture,
        pretrained  → summarize_layerwise_fit output
        random_init → summarize_layerwise_fit output
        init_stats  → summarize_initialization output
    """
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    pretrained = AutoModel.from_pretrained(model_id, trust_remote_code=False)
    rand_init  = AutoModel.from_config(config)

    try:
        pretrained_summary = summarize_layerwise_fit(
            collect_attention_tensors(pretrained, max_layers=max_layers),
            include_student_t=include_student_t,
        )
        random_summary = summarize_layerwise_fit(
            collect_attention_tensors(rand_init, max_layers=max_layers),
            include_student_t=include_student_t,
        )
        init_stats = summarize_initialization(rand_init)
    finally:
        # Release GPU memory deterministically for 4 GB VRAM environments.
        del pretrained, rand_init
        gc.collect()
        flush_cuda()

    return {
        "model_id":     model_id,
        "architecture": config.model_type,
        "pretrained":   pretrained_summary,
        "random_init":  random_summary,
        "init_stats":   init_stats,
    }


# ---------------------------------------------------------------------------
# Head-level weight collection  (where architecture exposes head dimension)
# ---------------------------------------------------------------------------

def collect_head_tensors(model: torch.nn.Module) -> Dict[int, List[np.ndarray]]:
    """Return ``{layer_idx: [head_0_flat, head_1_flat, …]}``.

    Handles both separate ``q_proj`` / ``k_proj`` / ``v_proj`` projections
    (BERT, RoBERTa, etc.) **and** fused projections such as GPT-2's
    ``c_attn`` by performing an in-place matrix decomposition when the
    architecture exposes ``num_attention_heads`` and ``hidden_size``.
    """
    result: Dict[int, List[np.ndarray]] = {}
    try:
        n_heads = model.config.num_attention_heads
        hidden_size = getattr(model.config, "hidden_size", None)
    except AttributeError:
        logger.warning("Model does not expose num_attention_heads; head-level analysis unavailable")
        return result

    head_dim: int | None = None
    fused_src = None  # (layer_idx, W_matrix, name) for fused case

    for name, param in model.named_parameters():
        if param.ndim < 2 or not name.endswith(".weight"):
            continue
        lowered = name.lower()
        # Separate projections
        if "q_proj" in lowered:
            head_dim = param.shape[0] // n_heads
            break
        # Fused projection (GPT-2 style c_attn)
        if "c_attn" in lowered and hidden_size is not None:
            if hidden_size % n_heads == 0:
                head_dim = hidden_size // n_heads
                match = _LAYER_PATTERN.search(lowered)
                if match:
                    fused_src = (int(match.group(2)), param.detach().cpu().numpy(), name)
                    break
            continue

    if head_dim is None:
        return result

    if fused_src is not None:
        layer_idx, W, src_name = fused_src
        # W shape is [3 * hidden_size, hidden_size]
        for h in range(n_heads):
            start, end = h * head_dim, (h + 1) * head_dim
            q_vec = W[start:end, :].reshape(-1)
            k_vec = W[hidden_size + start:hidden_size + end, :].reshape(-1)
            v_vec = W[2 * hidden_size + start:2 * hidden_size + end, :].reshape(-1)
            head_vec = np.concatenate([q_vec, k_vec, v_vec]).astype(np.float64)
            result.setdefault(layer_idx, []).append(head_vec)
        return result

    for name, param in model.named_parameters():
        if "q_proj" not in name or not name.endswith(".weight") or param.ndim < 2:
            continue
        match = _LAYER_PATTERN.search(name.lower())
        if not match:
            continue
        layer_idx = int(match.group(2))
        Wq = param.detach().cpu().numpy()          # [out, in]
        Wk_name = name.replace("query.weight", "key.weight")
        Wv_name = name.replace("query.weight", "value.weight")

        # locate matching k/v by param name lookup inside the module
        sd = model.state_dict()
        Wk = sd.get(Wk_name)
        Wv = sd.get(Wv_name)
        if Wk is None or Wv is None:
            continue

        Wq, Wk, Wv = Wq.detach().cpu().numpy(), Wk.detach().cpu().numpy(), Wv.detach().cpu().numpy()
        for h in range(n_heads):
            start, end = h * head_dim, (h + 1) * head_dim
            head_vec = np.concatenate([
                Wq[start:end, :].reshape(-1),
                Wk[start:end, :].reshape(-1),
                Wv[start:end, :].reshape(-1),
            ])
            result.setdefault(layer_idx, []).append(head_vec.astype(np.float64))

    return result
