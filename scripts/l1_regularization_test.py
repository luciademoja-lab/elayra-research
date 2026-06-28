"""
scripts/l1_regularization_test.py
===================================
Section 5.1 mechanistic hypothesis test.

Trains BERT-base with *random labels* under two conditions:
  (A) standard cross-entropy loss only (control)
  (B) CE loss + L1 penalty on all attention projection weight norms (treatment)

The hypothesis: if Laplace-distributed weights arise from implicit L1-
equivalent sparsity pressure, then explicit L1 regularisation should
increase Laplace prevalence even in a BERT-style (Gaussian-dominant)
architecture.

Usage
-----
    python scripts/l1_regularization_test.py
"""
from __future__ import annotations

import gc
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from ela.analysis import (
    MAX_LAYERS_PRIMARY,
    collect_attention_tensors,
    summarize_layerwise_fit,
)
from ela.config import L1RegularizationConfig
from ela.utils import build_batch, flush_cuda, seed_everything

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

cfg = L1RegularizationConfig()


def _attention_l1_norm(model: torch.nn.Module) -> torch.Tensor:
    """Sum of L1 norms of all attention projection weight tensors."""
    l1 = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if (param.ndim >= 2
                and name.endswith(".weight")
                and any(t in name.lower()
                        for t in ("query", "key", "value", "attn", "attention"))):
            l1 = l1 + param.norm(p=1)
    return l1


def _run(label: str, use_l1: bool) -> dict:
    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForMaskedLM.from_pretrained(cfg.model_id).to(device)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    before = summarize_layerwise_fit(
        collect_attention_tensors(model, max_layers=MAX_LAYERS_PRIMARY)
    )

    losses: list[float] = []
    for _ in tqdm(range(cfg.train_steps), desc=f"  {label}", leave=False):
        batch = {k: v.to(device) for k, v in build_batch(
            tokenizer, cfg.batch_size, cfg.seq_len, "masked").items()}
        out = model(**batch)
        ce_loss = out.loss
        total_loss = ce_loss + (cfg.l1_coeff * _attention_l1_norm(model) if use_l1 else 0.0)
        losses.append(float(ce_loss))
        total_loss.backward()
        optim.step()
        optim.zero_grad()

        del batch

    after = summarize_layerwise_fit(
        collect_attention_tensors(model, max_layers=MAX_LAYERS_PRIMARY)
    )

    del model, optim
    gc.collect()
    flush_cuda()

    return {
        "condition":   label,
        "use_l1":      use_l1,
        "before":      before,
        "after":       after,
        "final_ce_loss": losses[-1],
        "seed":        cfg.seed,
        "l1_coeff":    cfg.l1_coeff if use_l1 else 0.0,
    }


def main() -> None:
    log.info("=" * 80)
    log.info("L1 REGULARISATION TEST  (ela-backed)")
    log.info("Model=%s  Steps=%d  Batch=%d  L1_coeff=%g",
             cfg.model_id, cfg.train_steps, cfg.batch_size, cfg.l1_coeff)
    log.info("=" * 80)

    log.info("\n[Condition A] No L1 (control)…")
    ctrl = _run("no_l1", use_l1=False)
    log.info("  %.1f%% → %.1f%% Laplace  final_loss=%.4f",
             ctrl["before"]["laplace_pct"], ctrl["after"]["laplace_pct"], ctrl["final_ce_loss"])

    log.info("\n[Condition B] With L1 (treatment)…")
    treat = _run("l1_penalty", use_l1=True)
    log.info("  %.1f%% → %.1f%% Laplace  final_loss=%.4f",
             treat["before"]["laplace_pct"], treat["after"]["laplace_pct"], treat["final_ce_loss"])

    delta = (treat["after"]["laplace_pct"] - ctrl["after"]["laplace_pct"])
    log.info("\nΔ Laplace%% (treatment − control): %.2f", delta)
    log.info("Hypothesis %s: L1 increases Laplace%% in BERT",
             "SUPPORTED" if delta > 0 else "NOT SUPPORTED")

    out_path = os.path.join(os.path.dirname(__file__), "..", "results",
                            "l1_regularization_results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"control": ctrl, "treatment": treat, "delta_laplace_pct": delta},
                  fh, indent=2)
    log.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
