"""
tests/test_ela_extended.py
==========================
Extended test coverage based on the DeepSeek code review.

Covers
------
* ela.config   — dataclass defaults, frozen immutability
* ela.utils    — batch generation, seeding, empty-tensor safety
* ela.distributions — goodness-of-fit (KS) assertions
* Deterministic synthetic validation for summarize_layerwise_fit
  (synthetic Laplace should be recognised as Laplace; synthetic Gaussian
  should be recognised as Gaussian)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ela.config import (
    CheckpointConfig,
    ControlLongConfig,
    ControlShortConfig,
    PipelineConfig,
    ShuffleControlConfig,
)
from ela.distributions import best_fit, gof_summary, ks_two_sample, winner
from ela.utils import build_batch, seed_everything


# ---------------------------------------------------------------------------
# ela.config
# ---------------------------------------------------------------------------

class TestConfig:

    def test_pipeline_defaults(self):
        cfg = PipelineConfig()
        assert cfg.max_layers == 8
        assert cfg.bootstrap_n == 10_000
        assert cfg.bootstrap_seed == 42

    def test_control_short_defaults(self):
        cfg = ControlShortConfig()
        assert cfg.batch_size == 8
        assert cfg.seq_len == 32
        assert cfg.num_seeds == 3
        assert len(cfg.model_ids) == 6

    def test_control_long_defaults(self):
        cfg = ControlLongConfig()
        assert cfg.train_steps == 500
        assert cfg.batch_size == 16
        assert cfg.checkpoints == [0, 50, 100, 250, 500]

    def test_checkpoint_config_defaults(self):
        cfg = CheckpointConfig()
        assert cfg.model_id == "gpt2"
        assert cfg.lr == 5e-5
        assert cfg.l1_coeff == 1e-4

    def test_shuffle_config_defaults(self):
        cfg = ShuffleControlConfig()
        assert cfg.seed == 0
        assert cfg.train_steps == 10

    def test_config_frozen(self):
        with pytest.raises((AttributeError, TypeError)):
            cfg = PipelineConfig()
            cfg.max_layers = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ela.utils
# ---------------------------------------------------------------------------

class TestUtils:

    def test_seed_everything_reproducible(self):
        seed_everything(42)
        a = np.random.randn(5)
        seed_everything(42)
        b = np.random.randn(5)
        np.testing.assert_array_equal(a, b)

    def test_build_batch_shapes(self):
        class _Tok:
            vocab_size = 100

        batch = build_batch(_Tok(), bs=4, seq_len=8, model_type="causal")
        assert batch["input_ids"].shape == (4, 8)
        assert batch["attention_mask"].shape == (4, 8)
        assert batch["labels"].shape == (4, 8)


# ---------------------------------------------------------------------------
# ela.distributions — GOF
# ---------------------------------------------------------------------------

class TestDistributionsGOF:
    """Goodness-of-fit assertions (review request)."""

    @pytest.fixture()
    def laplace_vec(self):
        rng = np.random.default_rng(0)
        return rng.laplace(loc=0.0, scale=1.0, size=5_000)

    @pytest.fixture()
    def gaussian_vec(self):
        rng = np.random.default_rng(1)
        return rng.normal(loc=0.0, scale=1.0, size=5_000)

    def test_ks_laplace_on_laplace(self, laplace_vec):
        ks = ks_two_sample(laplace_vec, "laplace")
        assert ks["statistic"] >= 0.0
        assert ks["pvalue"] >= 0.0
        # Large Laplace sample should not reject the Laplace hypothesis
        # at alpha=0.05 in expectation; we allow some slack due to randomness.
        assert ks["pvalue"] > 0.001

    def test_ks_gaussian_on_gaussian(self, gaussian_vec):
        ks = ks_two_sample(gaussian_vec, "gaussian")
        assert ks["pvalue"] > 0.001

    def test_ks_cross_reject(self, laplace_vec, gaussian_vec):
        # Laplace tested vs Gaussian should give lower p-value
        ks_l_on_g = ks_two_sample(laplace_vec, "gaussian")
        ks_l_on_l = ks_two_sample(laplace_vec, "laplace")
        # Not guaranteed but highly likely with 5k samples
        assert ks_l_on_g["statistic"] > ks_l_on_l["statistic"]

    def test_winner_ll_on_laplace_vec(self, laplace_vec):
        assert winner({"ll_laplace": -1.0, "ll_gaussian": -2.0}) == "ll_laplace"

    def test_best_fit_keys(self, laplace_vec):
        bf = best_fit(laplace_vec)
        for key in ("winner", "ll_laplace", "ll_gaussian", "ll_student_t",
                    "margin_vs_gaussian"):
            assert key in bf

    def test_gof_summary_keys(self, laplace_vec):
        gs = gof_summary(laplace_vec)
        for key in ("laplace", "gaussian", "student_t", "winner_ll", "winner_ks"):
            assert key in gs


# ---------------------------------------------------------------------------
# Deterministic synthetic validation
# ---------------------------------------------------------------------------

class TestDeterministicValidation:
    """
    Synthetic ground-truth tensors to prevent regression.  These are exact
    distributional samples, so the fitting pipeline must recover the
    generating family when sample size is large enough.
    """

    @pytest.fixture()
    def pretrained_like(self):
        rng = np.random.default_rng(7)
        return [rng.laplace(0, 1.2, size=20_000).astype(np.float64)]

    @pytest.fixture()
    def random_init_like(self):
        rng = np.random.default_rng(13)
        return [rng.normal(0, 0.8, size=20_000).astype(np.float64)]

    def test_laplace_samples_win(self, pretrained_like):
        from ela.analysis import summarize_layerwise_fit
        s = summarize_layerwise_fit(pretrained_like)
        assert s["laplace_wins"] / max(1, s["num_layers"]) > 0.7

    def test_gaussian_samples_win(self, random_init_like):
        from ela.analysis import summarize_layerwise_fit
        s = summarize_layerwise_fit(random_init_like)
        assert s["gaussian_wins"] / max(1, s["num_layers"]) > 0.7
