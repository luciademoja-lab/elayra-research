"""
tests/test_analysis_core.py
============================
Smoke tests for the ela analysis primitives.

These tests are lightweight: they load gpt2 (the smallest model in the
registry), run the fit pipeline, and assert structural invariants.

Run with:
    python -m pytest tests/ -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pytest

# Ensure ela/ is importable when running tests from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ela.analysis import (
    MAX_LAYERS_PRIMARY,
    MAX_LAYERS_DEPTH,
    MODEL_IDS,
    analyze_model,
    collect_attention_tensors,
    collect_head_tensors,
    summarize_initialization,
    summarize_layerwise_fit,
)
from ela.bootstrap import bootstrap_ci
from ela.distributions import ll_all, winner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def gpt2_layers():
    """Load GPT-2 once per session and return per-layer tensors."""
    from transformers import GPT2Model
    model = GPT2Model.from_pretrained("gpt2")
    return collect_attention_tensors(model, max_layers=MAX_LAYERS_PRIMARY)


# ---------------------------------------------------------------------------
# collect_attention_tensors
# ---------------------------------------------------------------------------

class TestCollectAttentionTensors:

    def test_returns_list(self, gpt2_layers):
        assert isinstance(gpt2_layers, list)

    def test_non_empty(self, gpt2_layers):
        assert len(gpt2_layers) > 0

    def test_respects_max_layers(self):
        from transformers import GPT2Model
        model = GPT2Model.from_pretrained("gpt2")
        layers = collect_attention_tensors(model, max_layers=3)
        assert len(layers) <= 3

    def test_each_tensor_is_numpy(self, gpt2_layers):
        for t in gpt2_layers:
            assert isinstance(t, np.ndarray)

    def test_each_tensor_1d(self, gpt2_layers):
        for t in gpt2_layers:
            assert t.ndim == 1

    def test_each_tensor_float64(self, gpt2_layers):
        for t in gpt2_layers:
            assert t.dtype == np.float64

    def test_no_zeros_only(self, gpt2_layers):
        """At least some weights should be non-zero."""
        assert any(np.any(t != 0) for t in gpt2_layers)


# ---------------------------------------------------------------------------
# summarize_layerwise_fit
# ---------------------------------------------------------------------------

class TestSummarizeLayerwiseFit:

    def test_required_keys(self, gpt2_layers):
        result = summarize_layerwise_fit(gpt2_layers)
        for key in ("num_layers", "laplace_wins", "gaussian_wins",
                    "laplace_pct", "layers"):
            assert key in result

    def test_laplace_pct_range(self, gpt2_layers):
        result = summarize_layerwise_fit(gpt2_layers)
        assert 0.0 <= result["laplace_pct"] <= 100.0

    def test_win_counts_consistent(self, gpt2_layers):
        result = summarize_layerwise_fit(gpt2_layers)
        assert result["laplace_wins"] + result["gaussian_wins"] == result["num_layers"]

    def test_each_layer_has_ll(self, gpt2_layers):
        result = summarize_layerwise_fit(gpt2_layers)
        for layer in result["layers"]:
            assert "ll_laplace" in layer
            assert "ll_gaussian" in layer
            assert "better_fit" in layer
            assert layer["better_fit"] in ("Laplace", "Gaussian")

    def test_lls_are_finite(self, gpt2_layers):
        result = summarize_layerwise_fit(gpt2_layers)
        for layer in result["layers"]:
            assert np.isfinite(layer["ll_laplace"])
            assert np.isfinite(layer["ll_gaussian"])

    def test_winner_matches_lls(self, gpt2_layers):
        result = summarize_layerwise_fit(gpt2_layers)
        for layer in result["layers"]:
            expected = "Laplace" if layer["ll_laplace"] > layer["ll_gaussian"] else "Gaussian"
            assert layer["better_fit"] == expected


# ---------------------------------------------------------------------------
# summarize_initialization
# ---------------------------------------------------------------------------

class TestSummarizeInitialization:

    def test_returns_stats(self):
        from transformers import GPT2Config, GPT2Model
        model = GPT2Model(GPT2Config())
        stats = summarize_initialization(model)
        for key in ("mean", "std", "range", "kurtosis", "skewness"):
            assert key in stats

    def test_std_positive(self):
        from transformers import GPT2Config, GPT2Model
        model = GPT2Model(GPT2Config())
        stats = summarize_initialization(model)
        assert stats["std"] > 0

    def test_range_positive(self):
        from transformers import GPT2Config, GPT2Model
        model = GPT2Model(GPT2Config())
        stats = summarize_initialization(model)
        assert stats["range"] >= 0


# ---------------------------------------------------------------------------
# analyze_model (pretrained + random-init)
# ---------------------------------------------------------------------------

class TestAnalyzeModel:

    def test_returns_required_keys(self):
        result = analyze_model("gpt2", max_layers=3)
        for key in ("model_id", "architecture", "pretrained",
                    "random_init", "init_stats"):
            assert key in result

    def test_model_id_preserved(self):
        result = analyze_model("gpt2", max_layers=3)
        assert result["model_id"] == "gpt2"

    def test_pretrained_vs_random_both_valid(self):
        result = analyze_model("gpt2", max_layers=3)
        for sub in ("pretrained", "random_init"):
            assert 0.0 <= result[sub]["laplace_pct"] <= 100.0

    def test_init_stats_present(self):
        result = analyze_model("gpt2", max_layers=3)
        for key in ("mean", "std", "range", "kurtosis", "skewness"):
            assert key in result["init_stats"]


# ---------------------------------------------------------------------------
# ela/distributions
# ---------------------------------------------------------------------------

class TestDistributions:

    @pytest.fixture()
    def flat_vec(self):
        rng = np.random.default_rng(0)
        return rng.standard_normal(500).astype(np.float64)

    def test_ll_all_keys(self, flat_vec):
        ll = ll_all(flat_vec)
        for key in ("ll_laplace", "ll_gaussian", "ll_student_t"):
            assert key in ll

    def test_lls_finite(self, flat_vec):
        ll = ll_all(flat_vec)
        for v in ll.values():
            assert np.isfinite(v) and v < 0  # log-likelihoods are negative

    def test_winner_is_string(self, flat_vec):
        assert winner(ll_all(flat_vec)) in (
            "ll_laplace", "ll_gaussian", "ll_student_t"
        )


# ---------------------------------------------------------------------------
# ela/bootstrap
# ---------------------------------------------------------------------------

class TestBootstrap:

    def test_basic_ci(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(50)
        y = x + rng.normal(0, 0.5, 50)  # correlated
        ci = bootstrap_ci(x, y, n=500, seed=7)
        for key in ("stat", "ci_low", "ci_high", "n"):
            assert key in ci
        assert ci["n"] == 500
        assert ci["ci_low"] <= ci["ci_high"]

    def test_ci_not_nan(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(20)
        y = rng.standard_normal(20)
        ci = bootstrap_ci(x, y, n=200, seed=9)
        assert not np.isnan(ci["stat"])
        assert not np.isnan(ci["ci_low"])
        assert not np.isnan(ci["ci_high"])

    def test_raises_on_short_input(self):
        with pytest.raises(ValueError):
            bootstrap_ci([1.0], [2.0])


# ---------------------------------------------------------------------------
# ela/viz — smoke test (no display needed, Agg backend)
# ---------------------------------------------------------------------------

class TestViz:

    def test_layer_heatmap_saves(self):
        import matplotlib
        matplotlib.use("Agg")
        from ela.viz import layer_heatmap
        fake = [{"layers": [{"layer": 0, "better_fit": "Laplace"},
                            {"layer": 1, "better_fit": "Gaussian"}],
                 "model_id": "test-model"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = layer_heatmap(fake, out_path=os.path.join(tmpdir, "hm.png"))
            assert os.path.isfile(path)

    def test_bar_comparison_saves(self):
        import matplotlib
        matplotlib.use("Agg")
        from ela.viz import bar_comparison
        with tempfile.TemporaryDirectory() as tmpdir:
            path = bar_comparison(
                ["A", "B"], [3, 5], [2, 4],
                out_path=os.path.join(tmpdir, "bar.png"),
            )
            assert os.path.isfile(path)


# ---------------------------------------------------------------------------
# Model registry sanity
# ---------------------------------------------------------------------------

class TestModelRegistry:

    def test_model_ids_non_empty(self):
        assert len(MODEL_IDS) > 0

    def test_primary_contains_gpt2_bert(self):
        assert "gpt2" in MODEL_IDS
        assert "bert-base-uncased" in MODEL_IDS

    def test_optional_ids(self):
        from ela.analysis import OPTIONAL_MODEL_IDS
        assert isinstance(OPTIONAL_MODEL_IDS, list)
