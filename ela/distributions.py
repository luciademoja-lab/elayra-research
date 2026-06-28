"""
ela/distributions.py
====================
Thin wrappers around scipy.stats fitting so the rest of the codebase
never imports scipy directly.  Each function is an *atom* — a single,
testable piece of logic.

New: Student-t added as third candidate (Section 8 of the paper).
New: Kolmogorov-Smirnov goodness-of-fit assertions added per review.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Tuple

import numpy as np
from scipy.stats import kstest, laplace, norm, t as student_t

logger = logging.getLogger(__name__)

# Minimum density floor to avoid log(0)
_EPS = 1e-10

# Kolmogorov distribution survival approximation (for manual KS p-values).
# See Marsaglia, Tsang, Wang (2003) for the underlying CDF.
_KOLMOGOROV_EXP_COEFF = [
    (-2, 1.25167364568392822e-1),
    (-4, -1.39549942833341827e-1),
    (-6, 5.38881659110113024e-2),
    (-8, -1.19210089078630813e-2),
]


# ---------------------------------------------------------------------------
# Fit helpers
# ---------------------------------------------------------------------------

def fit_laplace(flat: np.ndarray) -> Tuple[float, float]:
    """Return (loc, scale) MLE for a Laplace distribution."""
    return tuple(laplace.fit(flat.astype(np.float64)))  # type: ignore[return-value]


def fit_gaussian(flat: np.ndarray) -> Tuple[float, float]:
    """Return (mean, std) MLE for a Normal distribution."""
    return tuple(norm.fit(flat.astype(np.float64)))  # type: ignore[return-value]


def fit_student_t(flat: np.ndarray) -> Tuple[float, float, float]:
    """Return (df, loc, scale) MLE for a Student's-t distribution."""
    return tuple(student_t.fit(flat.astype(np.float64)))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Log-likelihood helpers
# ---------------------------------------------------------------------------

def log_likelihood(flat: np.ndarray, dist_name: str) -> float:
    """Compute total log-likelihood of *flat* under the named distribution."""
    flat = flat.astype(np.float64)
    if dist_name == "laplace":
        loc, scale = fit_laplace(flat)
        pdf_vals = laplace.pdf(flat, loc=loc, scale=scale)
    elif dist_name == "gaussian":
        loc, scale = fit_gaussian(flat)
        pdf_vals = norm.pdf(flat, loc=loc, scale=scale)
    elif dist_name == "student_t":
        df, loc, scale = fit_student_t(flat)
        pdf_vals = student_t.pdf(flat, df=df, loc=loc, scale=scale)
    else:
        raise ValueError(f"Unknown distribution: {dist_name!r}")
    return float(np.sum(np.log(pdf_vals + _EPS)))


def ll_all(flat: np.ndarray) -> Dict[str, float]:
    """Log-likelihood under all three candidate distributions."""
    return {
        "ll_laplace":   log_likelihood(flat, "laplace"),
        "ll_gaussian":  log_likelihood(flat, "gaussian"),
        "ll_student_t": log_likelihood(flat, "student_t"),
    }


def winner(ll: Dict[str, float]) -> str:
    """Return the name of the distribution with the highest log-likelihood."""
    return max(ll, key=ll.__getitem__)  # type: ignore[return-value]


def best_fit(flat: np.ndarray) -> Dict[str, object]:
    """Fit all three distributions and return the winner + margin dict."""
    ll = ll_all(flat)
    w = winner(ll)
    return {
        "winner": w,
        "ll_laplace":   ll["ll_laplace"],
        "ll_gaussian":  ll["ll_gaussian"],
        "ll_student_t": ll["ll_student_t"],
        "margin_vs_gaussian": ll["ll_laplace"] - ll["ll_gaussian"],
    }


# ---------------------------------------------------------------------------
# Goodness-of-fit assertions  (review request)
# ---------------------------------------------------------------------------

def _ks_pvalue(D: float, n: int) -> float:
    """Survival function of the Kolmogorov distribution."""
    # Use the series expansion with partial fraction coefficients.
    x = (math.sqrt(n) + 0.25 + 0.75 / math.sqrt(n)) * D
    if x == 0:
        return 1.0
    s = 0.0
    for e, c in _KOLMOGOROV_EXP_COEFF:
        s += c * math.exp(e * x * x)
    return max(0.0, min(1.0, 2 * (1 - 0.5 * (1 + s))))


def ks_two_sample(flat: np.ndarray, dist_name: str = "laplace") -> Dict[str, float]:
    """Kolmogorov-Smirnov test comparing *flat* against the fitted distribution.

    Parameters
    ----------
    flat:
        Empirical sample.
    dist_name:
        One of ``"laplace"``, ``"gaussian"``, ``"student_t"``.

    Returns
    -------
    dict with ``statistic`` and ``pvalue``.
    """
    flat = flat.astype(np.float64)
    if dist_name == "laplace":
        loc, scale = fit_laplace(flat)
        stat, pvalue = kstest(flat, "laplace", args=(loc, scale))
    elif dist_name == "gaussian":
        loc, scale = fit_gaussian(flat)
        stat, pvalue = kstest(flat, "norm", args=(loc, scale))
    elif dist_name == "student_t":
        df, loc, scale = fit_student_t(flat)
        sorted_flat = np.sort(flat)
        n = len(sorted_flat)
        emp_cdf = np.arange(1, n + 1) / n
        theo_cdf = student_t.cdf(sorted_flat, df=df, loc=loc, scale=scale)
        stat = float(np.max(np.abs(emp_cdf - theo_cdf)))
        pvalue = _ks_pvalue(stat, n)
    else:
        raise ValueError(f"Unknown distribution: {dist_name!r}")
    return {"statistic": float(stat), "pvalue": float(pvalue)}


def gof_summary(flat: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Return KS test results for all three candidate distributions."""
    flat = flat.astype(np.float64)
    out: Dict[str, Dict[str, float]] = {}
    for name in ("laplace", "gaussian", "student_t"):
        ks = ks_two_sample(flat, name)
        ll = log_likelihood(flat, name)
        out[name] = {
            "ll": ll,
            "ks_statistic": ks["statistic"],
            "ks_pvalue": ks["pvalue"],
        }
    out["winner_ll"] = winner({k: v["ll"] for k, v in out.items()
                               if k in ("laplace", "gaussian", "student_t")})
    pvals = {k: v["ks_pvalue"]
             for k, v in out.items()
             if k in ("laplace", "gaussian", "student_t")}
    out["winner_ks"] = max(pvals, key=pvals.__getitem__)
    return out
