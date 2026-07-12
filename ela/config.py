"""
ela/config.py
=============
Centralised hyper-parameter defaults for the ela experimental suite.

Design goals
------------
* Replace the 5+ copies of ``BATCH_SIZE``, ``SEQ_LEN``, ``LR`` and ``SEED``
  scattered across ``scripts/`` with a single source of truth.
* Keep every value overridable so existing research behaviour is preserved
  unless the caller explicitly opts in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Pipeline defaults
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineConfig:
    """Defaults for the 15-model primary analysis."""

    max_layers: int = 8
    bootstrap_n: int = 10_000
    bootstrap_seed: int = 42
    bootstrap_alpha: float = 0.05


@dataclass(frozen=True)
class ControlShortConfig:
    """Defaults for the short-horizon multi-seed control."""

    model_ids: List[Tuple[str, str]] = field(
        default_factory=lambda: [
            ("gpt2", "causal"),
            ("gpt2-medium", "causal"),
            ("bert-base-uncased", "masked"),
            ("roberta-base", "masked"),
            ("facebook/bart-base", "seq2seq"),
            ("t5-small", "seq2seq"),
        ]
    )
    num_seeds: int = 3
    train_steps: int = 25
    batch_size: int = 8
    seq_len: int = 32
    lr: float = 5e-5
    seed: int = 42


@dataclass(frozen=True)
class ControlLongConfig:
    """Defaults for the 500-step long-horizon control."""

    model_ids: List[Tuple[str, str]] = field(
        default_factory=lambda: [
            ("gpt2", "causal"),
            ("bert-base-uncased", "masked"),
            ("google/electra-small-discriminator", "masked"),
        ]
    )
    train_steps: int = 500
    batch_size: int = 16
    seq_len: int = 32
    seed: int = 42
    checkpoints: List[int] = field(default_factory=lambda: [0, 50, 100, 250, 500])


@dataclass(frozen=True)
class CheckpointConfig:
    """Defaults for Section-8 checkpoint analysis (GPT-2 only)."""

    model_id: str = "gpt2"
    train_steps: int = 500
    checkpoint_every: int = 50
    batch_size: int = 8
    seq_len: int = 32
    lr: float = 5e-5
    l1_coeff: float = 1e-4
    seed: int = 42


@dataclass(frozen=True)
class ShuffleControlConfig:
    """Defaults for the random-label control."""

    model_ids: List[Tuple[str, str]] = field(
        default_factory=lambda: [
            ("gpt2", "causal"),
            ("bert-base-uncased", "masked"),
        ]
    )
    train_steps: int = 10
    batch_size: int = 4
    seq_len: int = 16
    lr: float = 5e-5
    seed: int = 0


@dataclass(frozen=True)
class L1RegularizationConfig:
    """Defaults for the L1 regularization hypothesis test (BERT-base)."""

    model_id: str = "bert-base-uncased"
    train_steps: int = 200
    batch_size: int = 8
    seq_len: int = 32
    lr: float = 5e-5
    l1_coeff: float = 1e-4
    seed: int = 42


# ---------------------------------------------------------------------------
# Backwards-compatible module-level aliases
# (scripts that previously used bare integers can migrate gradually)
# ---------------------------------------------------------------------------

DEFAULTS = ControlShortConfig()

BATCH_SIZE_SHORT   = DEFAULTS.batch_size
SEQ_LEN_SHORT      = DEFAULTS.seq_len
LR_SHORT           = DEFAULTS.lr
SEED_SHORT         = DEFAULTS.seed
NUM_SEEDS          = DEFAULTS.num_seeds
TRAIN_STEPS_SHORT  = DEFAULTS.train_steps

BATCH_SIZE_LONG    = ControlLongConfig().batch_size
SEQ_LEN_LONG       = ControlLongConfig().seq_len
SEED_LONG          = ControlLongConfig().seed
TRAIN_STEPS_LONG   = ControlLongConfig().train_steps
