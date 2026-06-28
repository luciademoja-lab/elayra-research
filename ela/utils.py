"""
ela/utils.py
============
Shared procedural helpers used across the ``scripts/`` entry points.

* ``build_batch`` — generates random token batches for control experiments,
  replacing 5+ duplicated inline implementations.
* ``flush_cuda`` — forces deterministic GPU cache release after a model has
  been processed (addresses 4 GB VRAM fragmentation risk).
"""

from __future__ import annotations

import logging
import random
from typing import Dict

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deterministic seeding helper
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Apply *seed* to torch, numpy and Python stdlib RNGs."""
    torch.manual_seed(seed)
    import numpy as _np  # local import keeps top-level imports lean
    _np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def build_batch(
    tokenizer,
    bs: int,
    seq_len: int,
    model_type: str = "causal",
) -> Dict[str, torch.Tensor]:
    """Return a dict of random-input tensors ready for ``model(**batch)``.

    Parameters
    ----------
    tokenizer:
        A ``transformers`` tokenizer exposing ``vocab_size``.
    bs:
        Batch size.
    seq_len:
        Sequence length.
    model_type:
        Architecture family (``"causal"`` or ``"masked"``).  Currently only
        affects the ``labels`` shape; both families share the same input
        structure here.

    Returns
    -------
    dict with ``input_ids``, ``attention_mask`` and ``labels``.
    """
    vocab = tokenizer.vocab_size
    ids = torch.randint(0, vocab, (bs, seq_len))
    labels = torch.randint(0, vocab, (bs, seq_len))
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# GPU memory hygiene
# ---------------------------------------------------------------------------

def flush_cuda() -> None:
    """Release cached GPU memory if a CUDA device is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
