"""
elayra-research — ela package
==============================
Analysis primitives for the Laplace-vs-Gaussian transformer weight study.

Architecture (atomic-habits style: each file does one thing):
  analysis.py   — model loading, weight collection, layer summarisation
  distributions — distribution fitting + log-likelihood helpers
  bootstrap.py  — bootstrap confidence intervals
  viz.py        — shared plotting helpers
"""

from ela.analysis import (
    collect_attention_tensors,
    summarize_layerwise_fit,
    summarize_initialization,
    analyze_model,
    MODEL_IDS,
    OPTIONAL_MODEL_IDS,
    CAPABLE_MODEL_IDS,
)

__all__ = [
    "collect_attention_tensors",
    "summarize_layerwise_fit",
    "summarize_initialization",
    "analyze_model",
    "MODEL_IDS",
    "OPTIONAL_MODEL_IDS",
    "CAPABLE_MODEL_IDS",
]
