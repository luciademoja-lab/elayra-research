"""
ela/viz.py
==========
Shared plotting helpers.

All figures are saved as PNG at 150 dpi.  A non-interactive backend is
selected automatically if DISPLAY is not available.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # noqa: E402 — must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_DPI = 150


def _save(fig, path: str, dpi: int = _DEFAULT_DPI) -> str:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved %s", path)
    return path


# ---------------------------------------------------------------------------
# Layer-wise heat-map  (model × layer → Laplace/Gaussian winner)
# ---------------------------------------------------------------------------

def layer_heatmap(
    results_list: Sequence[dict],
    title: str = "Layer-wise Laplace Dominance",
    figsize: tuple[int, int] = (14, 8),
    dpi: int = _DEFAULT_DPI,
    out_path: str = "layer_heatmap.png",
) -> str:
    """Plot a binary heatmap: 1 = Laplace wins, 0 = Gaussian wins."""
    model_ids = [r.get("model_id", str(i)) for i, r in enumerate(results_list)]
    max_layers = max(len(r.get("layers", [])) for r in results_list)
    grid = np.full((len(results_list), max_layers), np.nan)
    for i, r in enumerate(results_list):
        for layer in r.get("layers", []):
            grid[i, layer["layer"]] = 1.0 if layer["better_fit"] == "Laplace" else 0.0

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_yticks(range(len(model_ids)))
    ax.set_yticklabels(model_ids, fontsize=7)
    ax.set_xlabel("Layer index")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Gaussian ← → Laplace")
    return _save(fig, out_path, dpi)


# ---------------------------------------------------------------------------
# Depth-gradient curve  (Laplace% vs layer index, one line per model)
# ---------------------------------------------------------------------------

def depth_curves(
    results_list: Sequence[dict],
    models: Optional[List[str]] = None,
    title: str = "Laplace Prevalence by Depth",
    figsize: tuple[int, int] = (10, 6),
    dpi: int = _DEFAULT_DPI,
    out_path: str = "depth_curves.png",
) -> str:
    """Line plot of Laplace% per layer, grouped by model."""
    fig, ax = plt.subplots(figsize=figsize)
    for r in results_list:
        mid = r.get("model_id", "")
        if models and mid not in models:
            continue
        layers = r.get("layers", [])
        if not layers:
            continue
        x = [ln["layer"] for ln in layers]
        # cumulative Laplace% up to each layer
        laplace_flags = [1 if ln["better_fit"] == "Laplace" else 0 for ln in layers]
        cumulative_pct = np.cumsum(laplace_flags) / (np.arange(1, len(layers) + 1)) * 100
        ax.plot(x, cumulative_pct, marker="o", label=mid, linewidth=2)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Cumulative Laplace prevalence (%)")
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    return _save(fig, out_path, dpi)


# ---------------------------------------------------------------------------
# Side-by-side bar chart
# ---------------------------------------------------------------------------

def bar_comparison(
    labels: Sequence[str],
    laplace_vals: Sequence[int],
    gaussian_vals: Sequence[int],
    title: str = "Laplace vs Gaussian Wins",
    figsize: tuple[int, int] = (10, 5),
    dpi: int = _DEFAULT_DPI,
    out_path: str = "bar_comparison.png",
) -> str:
    """Grouped bar chart."""
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width / 2, laplace_vals, width, label="Laplace", color="crimson", alpha=0.75)
    ax.bar(x + width / 2, gaussian_vals, width, label="Gaussian", color="seagreen", alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Layer count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    return _save(fig, out_path, dpi)


# ---------------------------------------------------------------------------
# Initialisation kurtosis vs Laplace% scatter
# ---------------------------------------------------------------------------

def kurtosis_scatter(
    model_ids: Sequence[str],
    kurtosis_vals: Sequence[float],
    laplace_pcts: Sequence[float],
    rho: Optional[float] = None,
    figsize: tuple[int, int] = (8, 6),
    dpi: int = _DEFAULT_DPI,
    out_path: str = "kurtosis_scatter.png",
) -> str:
    """Scatter: init kurtosis → pretrained Laplace prevalence."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(kurtosis_vals, laplace_pcts, s=120, alpha=0.7, c="steelblue")
    for xi, yi, label in zip(kurtosis_vals, laplace_pcts, model_ids):
        ax.annotate(label, (xi, yi), fontsize=7, xytext=(4, 4), textcoords="offset points")
    if rho is not None:
        ax.set_title(f"Init Kurtosis vs Pretrained Laplace%  (Spearman ρ = {rho:.3f})")
    else:
        ax.set_title("Init Kurtosis vs Pretrained Laplace%")
    ax.set_xlabel("Initialisation kurtosis")
    ax.set_ylabel("Pretrained Laplace prevalence (%)")
    ax.grid(True, alpha=0.3)
    return _save(fig, out_path, dpi)


# ---------------------------------------------------------------------------
# Training trajectory (loss + Laplace% over steps)
# ---------------------------------------------------------------------------

def training_trajectory(
    step_labels: Sequence[int],
    laplace_pcts: Sequence[float],
    losses: Sequence[float],
    title: str = "Distributional Stability During Training",
    figsize: tuple[int, int] = (10, 5),
    dpi: int = _DEFAULT_DPI,
    out_path: str = "training_trajectory.png",
) -> str:
    """Two subplots: Laplace% over steps and loss over steps."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].plot(step_labels, laplace_pcts, "o-", color="crimson", linewidth=2)
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Laplace prevalence (%)")
    axes[0].set_title("Distributional regime")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(step_labels, losses, "s--", color="steelblue", linewidth=2)
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training loss (confirms genuine gradient updates)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12)
    return _save(fig, out_path, dpi)
