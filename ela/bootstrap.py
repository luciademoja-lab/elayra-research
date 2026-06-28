"""
ela/bootstrap.py
================
Bootstrap confidence-interval utilities.

Computes bootstrap 95% CIs for the Spearman rank correlation between
initialization kurtosis and pretrained Laplace% (Section 4.5 of the paper).
The pipeline reports ρ = 0.296 with CI95 [-0.27, 0.76].
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

_DEFAULT_N_BOOT = 10_000
_DEFAULT_SEED = 42


def bootstrap_ci(
    x: Sequence[float],
    y: Sequence[float],
    stat_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    n: int = _DEFAULT_N_BOOT,
    seed: int = _DEFAULT_SEED,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Return percentile bootstrap CI for *stat_fn* applied to *(x, y)*.

    Parameters
    ----------
    x, y:
        Paired sample arrays of equal length.
    stat_fn:
        A function ``(x_arr, y_arr) -> float``.  Defaults to Spearman ρ.
    n:
        Number of bootstrap replicates.
    seed:
        RNG seed for reproducibility.
    alpha:
        Significance level (0.05 → 95 % CI).

    Returns
    -------
    dict with keys:
        stat        – point estimate on the observed data
        ci_low      – lower bound
        ci_high     – upper bound
        n           – number of replicates used
    """
    rng = np.random.default_rng(seed)
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n_obs = len(x_arr)
    if n_obs != len(y_arr):
        raise ValueError(f"x and y length mismatch: {n_obs} vs {len(y_arr)}")
    if n_obs < 3:
        raise ValueError(f"Need ≥ 3 paired observations, got {n_obs}")

    if stat_fn is None:
        def stat_fn(a, b):  # type: ignore[misc]
            r, _ = spearmanr(a, b)
            return float(r)

    point = stat_fn(x_arr, y_arr)
    replicates = np.empty(n, dtype=float)
    for i in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        replicates[i] = stat_fn(x_arr[idx], y_arr[idx])

    lo = float(np.percentile(replicates, 100 * alpha / 2))
    hi = float(np.percentile(replicates, 100 * (1 - alpha / 2)))
    return {"stat": point, "ci_low": lo, "ci_high": hi, "n": n}


def bootstrap_mean(
    values: Sequence[float],
    n: int = _DEFAULT_N_BOOT,
    seed: int = _DEFAULT_SEED,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Bootstrap CI for the sample mean."""
    arr = np.asarray(values, dtype=float)
    return bootstrap_ci(arr, np.zeros_like(arr), stat_fn=np.mean, n=n, seed=seed, alpha=alpha)
