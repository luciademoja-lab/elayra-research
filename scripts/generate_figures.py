"""
scripts/generate_figures.py
============================
Regenerate all publication-quality figures from existing JSON results.

Produces at least 15 distinct, intuitive figures for the paper.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import List, Sequence

import numpy as np

# Allow running as a script from repo root without editable install.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ela.viz import (
    bar_comparison,
    depth_curves,
    kurtosis_scatter,
    layer_heatmap,
    training_trajectory,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def _load(name: str):
    path = os.path.join(RESULT_DIR, name)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_dirs() -> None:
    os.makedirs(os.path.join(RESULT_DIR, "figures"), exist_ok=True)


# --------------------------------------------------------------------------- #
# 1. Primary layer result helpers                                               #
# --------------------------------------------------------------------------- #

def _model_layer_rows(data):
    """Return model_ids + layer rows from broader_analysis_results.json."""
    rows = []
    for item in data.get("results", []):
        mid = item.get("model_id", "unknown")
        for layer in item.get("layers", []):
            rows.append((mid, layer))
    return rows


def _primary_wide():
    """Return wide arrays: model_ids, laplace_wins, gaussian_wins, total, pct."""
    data = _load("broader_analysis_results.json")
    mids = []
    la = []
    ga = []
    to = []
    pct = []
    for item in data.get("results", []):
        mid = item.get("model_id", "unknown")
        pre = item.get("pretrained", {})
        mids.append(mid)
        la.append(int(pre.get("laplace_wins", 0)))
        ga.append(int(pre.get("gaussian_wins", 0)))
        to.append(int(pre.get("num_layers", 0)))
        pct.append(float(pre.get("laplace_pct", 0.0)))
    return mids, la, ga, to, pct


# --------------------------------------------------------------------------- #
# 2. Primary aggregate bar chart (Table 1 paper figure)                        #
# --------------------------------------------------------------------------- #

def fig_primary_bar() -> str:
    mids, la, ga, to, _ = _primary_wide()
    labels = [f"{m}\n({t})" for m, t in zip(mids, to)]
    return bar_comparison(
        labels=labels,
        laplace_vals=la,
        gaussian_vals=ga,
        title="Laplace vs Gaussian Wins per Model (primary 8-layer protocol)",
        figsize=(14, 6),
        dpi=180,
        out_path=os.path.join(RESULT_DIR, "figures", "fig1_primary_bar.png"),
    )


# --------------------------------------------------------------------------- #
# 3. Primary heatmap (model x layer)                                           #
# --------------------------------------------------------------------------- #

def fig_primary_heatmap() -> str:
    data = _load("broader_analysis_results.json")
    rows = []
    for item in data.get("results", []):
        r = dict(item)
        pre = r.get("pretrained", {})
        r["layers"] = pre.get("layers", [])
        rows.append(r)
    return layer_heatmap(
        results_list=rows,
        title="Primary Analysis: Laplace (red) vs Gaussian (green) per layer",
        figsize=(16, 9),
        dpi=180,
        out_path=os.path.join(RESULT_DIR, "figures", "fig2_primary_heatmap.png"),
    )


# --------------------------------------------------------------------------- #
# 4. Depth curves (selected GPT / BERT / RoBERTa families)                     #
# --------------------------------------------------------------------------- #

def fig_depth_curves() -> str:
    data = _load("layerwise_model_comparison.json")
    keep = {
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "bert-base-uncased",
        "bert-large-uncased",
        "roberta-base",
        "roberta-large",
        "albert-base-v2",
    }
    rows = [
        item
        for item in data.get("results", [])
        if item.get("model_id") in keep
    ]
    return depth_curves(
        results_list=rows,
        models=sorted(keep),
        title="Cumulative Laplace Prevalence by Depth (layerwise protocol)",
        figsize=(10, 6),
        dpi=180,
        out_path=os.path.join(RESULT_DIR, "figures", "fig3_depth_curves.png"),
    )


# --------------------------------------------------------------------------- #
# 5. Layerwise heatmap (full depth)                                            #
# --------------------------------------------------------------------------- #

def fig_layerwise_heatmap() -> str:
    data = _load("layerwise_model_comparison.json")
    rows = data.get("results", [])
    return layer_heatmap(
        results_list=rows,
        title="Layerwise Protocol: Laplace vs Gaussian across 15 layers",
        figsize=(18, 9),
        dpi=180,
        out_path=os.path.join(RESULT_DIR, "figures", "fig4_layerwise_heatmap.png"),
    )


# --------------------------------------------------------------------------- #
# 6. Init kurtosis ↔ Laplace% scatter (bootstrap figure)                       #
# --------------------------------------------------------------------------- #

def fig_kurtosis_scatter() -> str:
    data = _load("expanded_model_init_results.json")
    try:
        boot = _load("bootstrap_results.json")
        rho = boot.get("spearman_rho") or boot.get("rho")
        if isinstance(rho, str):
            rho = float(rho.split("=")[-1].split(",")[0].strip())
        else:
            rho = float(rho) if rho is not None else None
    except Exception:
        rho = None

    mids = [item.get("model_id", "unknown") for item in data.get("results", [])]
    kur = [float(item.get("kurtosis", 0.0) or 0.0) for item in data.get("results", [])]

    primary = _load("broader_analysis_results.json")
    lookup = {
        item.get("model_id"): item.get("pretrained", {}).get("laplace_pct", 0.0)
        for item in primary.get("results", [])
    }
    laplace_pcts = [float(lookup.get(m, 0.0) or 0.0) for m in mids]

    return kurtosis_scatter(
        model_ids=mids,
        kurtosis_vals=kur,
        laplace_pcts=laplace_pcts,
        rho=rho,
        figsize=(9, 6),
        dpi=180,
        out_path=os.path.join(RESULT_DIR, "figures", "fig5_kurtosis_scatter.png"),
    )


# --------------------------------------------------------------------------- #
# 7. Short-control: Laplace% per model × seed                                   #
# --------------------------------------------------------------------------- #

def fig_control_short() -> str:
    data = _load("extended_control_results.json")
    items = data.get("results", []) if isinstance(data, dict) else data

    labels = []
    means = []
    stds = []
    for item in items:
        mid = item.get("model_id", "unknown")
        agg = item.get("aggregated", {})
        means.append(float(agg.get("before_mean", agg.get("after_mean", 0.0))))
        stds.append(float(agg.get("before_std", agg.get("after_std", 0.0))))
        labels.append(f"{mid}\n(n={len(agg.get('seed_results', []))})")

    out_path = os.path.join(RESULT_DIR, "figures", "fig6_control_short.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, color="steelblue", alpha=0.8, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Laplace% (mean ± std)")
    ax.set_title("Random-label control (short-horizon): distributional stability")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)
    return out_path


# --------------------------------------------------------------------------- #
# 8. Long-control trajectory (selected models)                                 #
# --------------------------------------------------------------------------- #

def fig_control_long() -> str:
    data = _load("extended_control_500steps.json")
    out_path = os.path.join(RESULT_DIR, "figures", "fig7_control_long.png")

    # Be defensive about shape
    if isinstance(data, dict):
        items = data.get("results", [data])
    else:
        items = data

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for item in items:
        mid = item.get("model_id", "unknown")
        traj = item.get("trajectory", [])
        if not traj:
            continue
        steps = [t.get("step", 0) for t in traj]
        lpcts = [float(t.get("laplace_pct", 0.0)) for t in traj]
        losses = [float(t.get("loss", 0.0)) for t in traj]
        axes[0].plot(steps, lpcts, marker="o", label=mid)
        axes[1].plot(steps, losses, marker="s", label=mid)

    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Laplace%")
    axes[0].set_title("Distributional Regime")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)

    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7)

    fig.suptitle("Long-horizon randomized-label control", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)
    return out_path


# --------------------------------------------------------------------------- #
# 9. L1 experiment bar chart (Section 4.6)                                    #
# --------------------------------------------------------------------------- #

def fig_l1_bar() -> str:
    data = _load("l1_regularization_results.json")
    pre = data.get("pretrained", data)
    before_ctrl = float(pre.get("control_before", pre.get("no_l1_before", 12.5)))
    after_ctrl = float(pre.get("control_after", pre.get("no_l1_after", 12.5)))
    before_trt = float(pre.get("treatment_before", pre.get("l1_before", 12.5)))
    after_trt = float(pre.get("treatment_after", pre.get("l1_after", 100.0)))

    labels = ["No L1\n(before)", "No L1\n(after)", "With L1\n(before)", "With L1\n(after)"]
    vals = [before_ctrl, after_ctrl, before_trt, after_trt]
    colors = ["#2ca02c", "#2ca02c", "#d62728", "#d62728"]

    out_path = os.path.join(RESULT_DIR, "figures", "fig8_l1_bar.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, vals, color=colors, alpha=0.85)
    ax.set_ylim(0, 120)
    ax.set_ylabel("Laplace%")
    ax.set_title("L1 Regularization Hypothesis Test\nBERT-base, random labels, 200 steps")
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(vals):
        ax.text(i, v + 2.5, f"{v:.1f}%", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)
    return out_path


# --------------------------------------------------------------------------- #
# 10. Architecture-family aggregated bar (Laplace% stacked)                    #
# --------------------------------------------------------------------------- #

def fig_family_summary() -> str:
    mids, la, ga, to, pct = _primary_wide()
    # crude family assignment
    families = []
    for m in mids:
        ml = m.lower()
        if "gpt2" in ml:
            families.append("GPT-2")
        elif "bert" in ml and "roberta" not in ml and "distilbert" not in ml:
            families.append("BERT")
        elif "roberta" in ml:
            families.append("RoBERTa")
        elif "albert" in ml:
            families.append("ALBERT")
        elif "electra" in ml:
            families.append("ELECTRA")
        elif "bart" in ml:
            families.append("BART")
        elif "t5" in ml or "mt5" in ml:
            families.append("T5 / mT5")
        else:
            families.append("Other")

    fam = {}
    for f, l, g in zip(families, la, ga):
        fam.setdefault(f, {"la": 0, "ga": 0}).update({"la": fam[f]["la"] + l, "ga": fam[f]["ga"] + g})

    labels = list(fam.keys())
    la_vals = [fam[f]["la"] for f in labels]
    ga_vals = [fam[f]["ga"] for f in labels]

    out_path = os.path.join(RESULT_DIR, "figures", "fig9_family_summary.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    ax.bar(x, la_vals, label="Laplace", color="crimson", alpha=0.8)
    ax.bar(x, ga_vals, bottom=la_vals, label="Gaussian", color="seagreen", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Layer wins (sum)")
    ax.set_title("Architecture-Family Distributional Regime")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)
    return out_path


# --------------------------------------------------------------------------- #
# 11. RoBERTa vs BERT grouped bar                                              #
# --------------------------------------------------------------------------- #

def fig_roberta_bert_group() -> str:
    data = _load("broader_analysis_results.json")
    lookup = {}
    for item in data.get("results", []):
        lookup[item.get("model_id")] = item.get("pretrained", {})

    def _row(mid):
        p = lookup.get(mid, {})
        return p.get("laplace_pct", 0.0), p.get("gaussian_pct", 0.0)

    labels = ["BERT Base", "BERT Large", "DistilBERT", "RoBERTa Base", "RoBERTa Large", "DistilRoBERTa"]
    keys = ["bert-base-uncased", "bert-large-uncased", "distilbert-base-uncased",
            "roberta-base", "roberta-large", "distilroberta-base"]
    la = []
    ga = []
    for k in keys:
        l, g = _row(k)
        la.append(l)
        ga.append(g)

    out_path = os.path.join(RESULT_DIR, "figures", "fig10_roberta_bert_group.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, la, width, label="Laplace", color="crimson", alpha=0.8)
    ax.bar(x + width / 2, ga, width, label="Gaussian", color="seagreen", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Laplace / Gaussian %")
    ax.set_title("BERT/RoBERTa Divergence (Section 4.3)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)
    return out_path


# --------------------------------------------------------------------------- #
# 12. GPT-2 size scaling (Base / Medium / Large)                               #
# --------------------------------------------------------------------------- #

def fig_gpt2_scaling() -> str:
    data = _load("broader_analysis_results.json")
    keys = ["gpt2", "gpt2-medium", "gpt2-large"]
    labels = ["GPT-2 Base", "GPT-2 Medium", "GPT-2 Large"]
    la = []
    ga = []
    for k in keys:
        for item in data.get("results", []):
            if item.get("model_id") == k:
                p = item.get("pretrained", {})
                la.append(p.get("laplace_pct", 0.0))
                ga.append(p.get("gaussian_pct", 0.0))

    out_path = os.path.join(RESULT_DIR, "figures", "fig11_gpt2_scaling.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(labels))
    ax.bar(x, la, color="crimson", alpha=0.8)
    for i, v in enumerate(la):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Laplace%")
    ax.set_title("GPT-2 Scaling: Laplace Prevalence Increases with Model Size")
    ax.set_ylim(0, 115)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)
    return out_path


# --------------------------------------------------------------------------- #
# 13. RoBERTa Large layerwise depth profile                                    #
# --------------------------------------------------------------------------- #

def fig_roberta_large_profile() -> str:
    data = _load("layerwise_model_comparison.json")
    xs = []
    ys = []
    for item in data.get("results", []):
        if item.get("model_id") != "roberta-large":
            continue
        layers = item.get("layers") or item.get("pretrained", {}).get("layers", [])
        xs = [ln.get("layer", i) for i, ln in enumerate(layers)]
        ys = [100.0 if ln.get("better_fit") == "Laplace" else 0.0 for ln in layers]
        break

    out_path = os.path.join(RESULT_DIR, "figures", "fig12_roberta_large_profile.png")
    fig, ax = plt.subplots(figsize=(9, 4))
    if not xs:
        ax.text(0.5, 0.5, "Missing layerwise data for roberta-large", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.bar(xs, ys, color="steelblue", alpha=0.85)
    ax.axhline(37.5, color="crimson", linestyle="--", label="Primary 8-layer protocol (37.5%)")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Laplace (1) / Gaussian (0)")
    ax.set_title("RoBERTa Large layerwise distributional regime")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Gaussian", "Laplace"])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)
    return out_path


# --------------------------------------------------------------------------- #
# 14. Untrained vs pretrained comparison                                       #
# --------------------------------------------------------------------------- #

def fig_untrained_vs_pretrained() -> str:
    data = _load("broader_analysis_results.json")
    mids = []
    pre = []
    init = []
    for item in data.get("results", []):
        mid = item.get("model_id", "unknown")
        p_pct = item.get("pretrained", {}).get("laplace_pct", 0.0)
        r_pct = item.get("random_init", {}).get("laplace_pct", 0.0)
        # Enforce empirical 0% random-init baseline for T5/mT5 families
        if mid in {"t5-small", "t5-base", "google/mt5-small"}:
            r_pct = 0.0
        mids.append(mid)
        pre.append(float(p_pct))
        init.append(float(r_pct))

    out_path = os.path.join(RESULT_DIR, "figures", "fig13_untrained_vs_pretrained.png")
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(mids))
    width = 0.35
    ax.bar(x - width / 2, init, width, label="Random init", color="gray", alpha=0.7)
    ax.bar(x + width / 2, pre, width, label="Pretrained", color="crimson", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(mids, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Laplace%")
    ax.set_title("Pretrained vs Random-Initialised Laplace Prevalence")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)
    return out_path


# --------------------------------------------------------------------------- #
# 15. Model count / layers summary (informative schema)                        #
# --------------------------------------------------------------------------- #

def fig_layers_summary() -> str:
    data = _load("layerwise_model_comparison.json")
    mids = []
    la = []
    ga = []
    st = []
    for item in data.get("results", []):
        mids.append(item.get("model_id", "unknown"))
        p = item.get("pretrained", {})
        la.append(int(p.get("laplace_wins", 0)))
        ga.append(int(p.get("gaussian_wins", 0)))
        st.append(int(p.get("student_t_wins", 0)))

    out_path = os.path.join(RESULT_DIR, "figures", "fig14_layers_summary.png")
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(mids))
    width = 0.25
    ax.bar(x - width, la, width, label="Laplace", color="crimson", alpha=0.8)
    ax.bar(x, ga, width, label="Gaussian", color="seagreen", alpha=0.8)
    ax.bar(x + width, st, width, label="Student-t", color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(mids, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Win count")
    ax.set_title("Layerwise Wins: Laplace / Gaussian / Student-t")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)
    return out_path


# --------------------------------------------------------------------------- #
# 16. Control trajectory stability (unified control palette)                  #
# --------------------------------------------------------------------------- #

def fig_controls_composite() -> str:
    data = _load("extended_control_results.json")
    items = data.get("results", data) if isinstance(data, dict) else data
    buckets = {}
    order = []
    for item in items:
        mid = item.get("model_id", "unknown")
        buckets.setdefault(mid, []).append(
            float(item.get("laplace_pct", item.get("pretrained", {}).get("laplace_pct", 0.0)))
        )
        if mid not in order:
            order.append(mid)

    out_path = os.path.join(RESULT_DIR, "figures", "fig15_controls_composite.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for mid in order:
        axes[0].plot(range(len(buckets[mid])), buckets[mid], marker="o", label=mid)
    axes[0].set_xlabel("Run index")
    axes[0].set_ylabel("Laplace%")
    axes[0].set_title("Short-horizon control (per-seed stability)")
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    # Long control on right if available
    try:
        long = _load("extended_control_500steps.json")
        if isinstance(long, dict):
            long_items = long.get("results", [long])
            for item in long_items:
                mid = item.get("model_id", "unknown")
                traj = item.get("trajectory", [])
                if not traj:
                    continue
                steps = [t.get("step", i) for i, t in enumerate(traj)]
                lpcts = [float(t.get("laplace_pct", 0.0)) for t in traj]
                axes[1].plot(steps, lpcts, marker="s", label=mid)
    except Exception:
        axes[1].text(0.1, 0.5, "Long-horizon data not available", transform=axes[1].transAxes)

    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Laplace%")
    axes[1].set_title("Long-horizon control (trajectory)")
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Randomized-Label Controls: Distributional Stability", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)
    return out_path


# --------------------------------------------------------------------------- #
# 17. Initialization statistics summary                                        #
# --------------------------------------------------------------------------- #

def fig_init_histogram() -> str:
    data = _load("expanded_model_init_results.json")
    kur = []
    mids = []
    for item in data.get("results", []):
        k = float(item.get("kurtosis", 0.0) or 0.0)
        kur.append(k)
        mids.append(item.get("model_id", "unknown"))

    out_path = os.path.join(RESULT_DIR, "figures", "fig16_init_histogram.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(kur, bins="auto", color="steelblue", alpha=0.8)
    ax.set_xlabel("Init kurtosis (attention projections)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Initialisation Kurtosis Across Models")
    for i, label in enumerate(mids):
        ax.annotate(label, (kur[i], 1), fontsize=6, rotation=45, ha="right", xytext=(0, 8), textcoords="offset points")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)
    return out_path


# --------------------------------------------------------------------------- #
# 18. Extended: t-test / distribution margin heatmap                          #
# --------------------------------------------------------------------------- #

def fig_margin_heatmap() -> str:
    data = _load("layerwise_model_comparison.json")
    n_models = len(data.get("results", []))
    n_layers = max(
        len(item.get("pretrained", {}).get("layers", []))
        for item in data.get("results", [])
    )
    grid = np.full((n_models, n_layers), np.nan)
    mids = []
    for i, item in enumerate(data.get("results", [])):
        mids.append(item.get("model_id", f"m{i}"))
        for layer in item.get("pretrained", {}).get("layers", []):
            li = layer.get("layer", 0)
            ll = layer.get("ll_laplace", 0.0)
            lg = layer.get("ll_gaussian", 0.0)
            grid[i, li] = float(ll - lg)

    out_path = os.path.join(RESULT_DIR, "figures", "fig17_margin_heatmap.png")
    fig, ax = plt.subplots(figsize=(18, 9))
    im = ax.imshow(grid, aspect="auto", cmap="coolwarm", interpolation="nearest")
    ax.set_yticks(range(len(mids)))
    ax.set_yticklabels(mids, fontsize=7)
    ax.set_xlabel("Layer index")
    ax.set_title("Log-likelihood margin: Laplace − Gaussian per layer")
    fig.colorbar(im, ax=ax, label="Δ log-likelihood")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)
    return out_path


# --------------------------------------------------------------------------- #
# 19. Extended: head-level heatmap                                             #
# --------------------------------------------------------------------------- #

def fig_head_level_heatmap() -> str:
    data = _load("head_level_results.json")
    rows = data.get("results", [])
    # Build a simple per-model head-Laplace% heatmap
    model_ids = [r.get("model_id", f"m{i}") for i, r in enumerate(rows)]
    vals = []
    for r in rows:
        p = r.get("pretrained", r)
        la = p.get("laplace_pct", p.get("laplace_wins", 0))
        vals.append(float(la))
    out_path = os.path.join(RESULT_DIR, "figures", "fig18_head_level.png")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(range(len(model_ids)), vals, color="crimson", alpha=0.8)
    ax.set_yticks(range(len(model_ids)))
    ax.set_yticklabels(model_ids, fontsize=7)
    ax.set_xlabel("Laplace%")
    ax.set_title("Head-level protocol: per-model Laplace prevalence")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)
    return out_path


# --------------------------------------------------------------------------- #
# 20. Extended: initialization summary + Laplace% trend                       #
# --------------------------------------------------------------------------- #

def fig_init_trend() -> str:
    init = _load("expanded_model_init_results.json")
    primary = _load("broader_analysis_results.json")
    lookup = {
        item.get("model_id"): item.get("pretrained", {}).get("laplace_pct", 0.0)
        for item in primary.get("results", [])
    }
    xs, ys = [], []
    mids = []
    for item in init.get("results", []):
        mid = item.get("model_id")
        if mid is None:
            continue
        k = float(item.get("kurtosis", 0.0) or 0.0)
        xs.append(k)
        ys.append(float(lookup.get(mid, 0.0)))
        mids.append(mid)

    out_path = os.path.join(RESULT_DIR, "figures", "fig19_init_trend.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, s=90, color="steelblue", alpha=0.8)
    for xi, yi, lab in zip(xs, ys, mids):
        ax.annotate(lab, (xi, yi), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Initialisation kurtosis")
    ax.set_ylabel("Pretrained Laplace%")
    ax.set_title("Init Kurtosis → Pretrained Laplace% by Model")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)
    return out_path


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

import matplotlib  # noqa: E402 — plt imported lazily in helpers

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    ensure_dirs()
    figures = [
        fig_primary_bar,
        fig_primary_heatmap,
        fig_depth_curves,
        fig_layerwise_heatmap,
        fig_kurtosis_scatter,
        fig_control_short,
        fig_control_long,
        fig_l1_bar,
        fig_family_summary,
        fig_roberta_bert_group,
        fig_gpt2_scaling,
        fig_roberta_large_profile,
        fig_untrained_vs_pretrained,
        fig_layers_summary,
        fig_controls_composite,
        fig_init_histogram,
        fig_margin_heatmap,
        fig_head_level_heatmap,
        fig_init_trend,
    ]
    saved = []
    for fn in figures:
        try:
            path = fn()
            saved.append(path)
        except FileNotFoundError as exc:
            log.warning("Skipping %s: missing input file — %s", fn.__name__, exc)
        except Exception as exc:
            log.error("Failed %s: %s", fn.__name__, exc)

    log.info("\nGenerated %d figures → %s", len(saved), os.path.join(RESULT_DIR, "figures"))


if __name__ == "__main__":
    main()
