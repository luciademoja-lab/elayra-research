"""
ela/distributions.py
====================
Thin wrappers around scipy.stats fitting so the rest of the codebase
never imports scipy directly.  Each function is an *atom* — a single,
testable piece of logic.

New: Student-t added as third candidate (Section 8 of the paper).
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
from scipy.stats import laplace, norm, t as student_t

logger = logging.getLogger(__name__)

# Minimum density floor to avoid log(0)
_EPS = 1e-10


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
