"""
elayra-research — ela package
==============================
Analysis primitives for the Laplace-vs-Gaussian transformer weight study.

Architecture (atomic-habits style: each file does one thing):
  analysis.py   — model loading, weight collection, layer summarisation
  bootstrap.py  — bootstrap confidence intervals
  config.py     — centralised hyper-parameter defaults
  distributions — distribution fitting + log-likelihood helpers
  utils.py      — batch generation, seeding, GPU memory hygiene
  viz.py        — shared plotting helpers
"""

from ela.analysis import (
    analyze_model,
    CAPABLE_MODEL_IDS,
    collect_attention_tensors,
    collect_head_tensors,
    MODEL_IDS,
    OPTIONAL_MODEL_IDS,
    summarize_initialization,
    summarize_layerwise_fit,
)

__all__ = [
    "analyze_model",
    "CAPABLE_MODEL_IDS",
    "collect_attention_tensors",
    "collect_head_tensors",
    "MODEL_IDS",
    "OPTIONAL_MODEL_IDS",
    "summarize_initialization",
    "summarize_layerwise_fit",
]
