"""
scripts/control_short.py
========================
Replacement for extended_control_experiment.py.

Short-horizon multi-seed control: 6 models × 3 seeds × 25 random-label
training steps.  Confirms that architecture alone determines the
distributional regime (Section 4.5 of the paper).

Usage
-----
    python scripts/control_short.py
"""
from __future__ import annotations

import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

from ela.analysis import (
    ATTENTION_TOKENS,
    MAX_LAYERS_PRIMARY,
    collect_attention_tensors,
    summarize_layerwise_fit,
)
from ela.viz import training_trajectory

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

CONTROL_MODELS = [
    ("gpt2",               "causal"),
    ("gpt2-medium",        "causal"),
    ("bert-base-uncased",  "masked"),
    ("roberta-base",       "masked"),
    ("facebook/bart-base", "causal"),
    ("t5-small",           "causal"),
]

NUM_SEEDS    = 3
TRAIN_STEPS  = 25
BATCH_SIZE   = 8
SEQ_LEN      = 32


def _build_batch(tokenizer, bs: int) -> dict:
    vocab = tokenizer.vocab_size
    ids = torch.randint(0, vocab, (bs, SEQ_LEN))
    return {
        "input_ids":      ids,
        "attention_mask": torch.ones_like(ids),
        "labels":         torch.randint(0, vocab, (bs, SEQ_LEN)),
    }


def _run_seed(model_id: str, model_type: str, seed: int) -> dict:
    torch.manual_seed(seed)
    import random; random.seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    ctor = (AutoModelForCausalLM if model_type == "causal"
            else AutoModelForMaskedLM)
    model = ctor.from_pretrained(model_id).to(device)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=5e-5)

    before = summarize_layerwise_fit(
        collect_attention_tensors(model, max_layers=MAX_LAYERS_PRIMARY)
    )

    losses: list[float] = []
    for _ in range(TRAIN_STEPS):
        batch = {k: v.to(device) for k, v in _build_batch(tokenizer, BATCH_SIZE).items()}
        out = model(**batch)
        loss = out.loss
        losses.append(float(loss))
        loss.backward()
        optim.step()
        optim.zero_grad()

    after = summarize_layerwise_fit(
        collect_attention_tensors(model, max_layers=MAX_LAYERS_PRIMARY)
    )
    return {"seed": seed, "before": before, "after": after, "final_loss": losses[-1]}


def _aggregate(seed_results: list) -> dict:
    b = [r["before"]["laplace_pct"] for r in seed_results]
    a = [r["after"]["laplace_pct"]  for r in seed_results]
    return {
        "before_mean": float(np.mean(b)),
        "before_std":  float(np.std(b)),
        "after_mean":  float(np.mean(a)),
        "after_std":   float(np.std(a)),
        "change_mean": float(np.mean([x - y for x, y in zip(a, b)])),
        "seed_results": seed_results,
    }


def main() -> None:
    log.info("=" * 80)
    log.info("SHORT-HORIZON MULTI-SEED CONTROL  (ela-backed)")
    log.info("Models=%d  Seeds=%d  Steps=%d  Batch=%d",
             len(CONTROL_MODELS), NUM_SEEDS, TRAIN_STEPS, BATCH_SIZE)
    log.info("=" * 80)

    all_out = []
    for mid, mtype in CONTROL_MODELS:
        try:
            log.info("\n[%s] (%s)", mid, mtype)
            seeds = [_run_seed(mid, mtype, s) for s in range(NUM_SEEDS)]
            agg = _aggregate(seeds)
            all_out.append({"model_id": mid, "model_type": mtype, "aggregated": agg})
            for sr in seeds:
                log.info("  seed %d: %.1f%% → %.1f%%  (loss=%.4f)",
                         sr["seed"], sr["before"]["laplace_pct"],
                         sr["after"]["laplace_pct"], sr["final_loss"])
            log.info("  mean: %.1f (±%.1f) → %.1f (±%.1f)  Δ %.2f",
                     agg["before_mean"], agg["before_std"],
                     agg["after_mean"],  agg["after_std"],
                     agg["change_mean"])
        except Exception as exc:
            log.info("  FAILED: %s", exc)

    out_path = os.path.join(os.path.dirname(__file__), "..", "results",
                            "extended_control_results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"results": all_out}, fh, indent=2)
    log.info("\nSaved → %s", out_path)


if __name__ == "__main__":
    main()
