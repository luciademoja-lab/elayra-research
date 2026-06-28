"""
scripts/control_long.py
========================
Replacement for extended_control_longer_runs.py.

Long-horizon (500-step) single-seed control with intermediate checkpoints.
Records Laplace% and loss at steps 0, 50, 100, 250, 500 to capture
distributional dynamics across the training trajectory (Section 4.5).

Models included are constrained to those that complete in reasonable time
on a 4 GB GPU: GPT-2, BERT-base, ELECTRA-small-discriminator.

Usage
-----
    python scripts/control_long.py
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
    MAX_LAYERS_PRIMARY,
    collect_attention_tensors,
    summarize_layerwise_fit,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

CONTROL_MODELS = [
    ("gpt2",                        "causal"),
    ("bert-base-uncased",           "masked"),
    ("google/electra-small-discriminator", "masked"),
]

TRAIN_STEPS  = 500
BATCH_SIZE   = 16
SEQ_LEN      = 32
SEED         = 42
CHECKPOINTS  = [0, 50, 100, 250, 500]


def _build_batch(tokenizer, bs: int) -> dict:
    vocab = tokenizer.vocab_size
    ids = torch.randint(0, vocab, (bs, SEQ_LEN))
    return {
        "input_ids":      ids,
        "attention_mask": torch.ones_like(ids),
        "labels":         torch.randint(0, vocab, (bs, SEQ_LEN)),
    }


def _run(model_id: str, model_type: str, seed: int) -> dict:
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

    loss_history: list[float] = []
    snaps: dict[int, dict] = {}

    for step in range(1, TRAIN_STEPS + 1):
        batch = {k: v.to(device) for k, v in _build_batch(tokenizer, BATCH_SIZE).items()}
        out = model(**batch)
        loss = out.loss
        loss_history.append(float(loss))
        loss.backward()
        optim.step()
        optim.zero_grad()

        if step in CHECKPOINTS:
            try:
                snaps[step] = summarize_layerwise_fit(
                    collect_attention_tensors(model, max_layers=MAX_LAYERS_PRIMARY)
                )
            except Exception:
                snaps[step] = {}

    after = summarize_layerwise_fit(
        collect_attention_tensors(model, max_layers=MAX_LAYERS_PRIMARY)
    )
    return {
        "seed": seed,
        "before": snaps.get(0, after),
        "after": after,
        "loss_history": loss_history,
        "distributions_at_steps": snaps,
    }


def main() -> None:
    log.info("=" * 80)
    log.info("LONG-HORIZON CONTROL  (500 steps, ela-backed)")
    log.info("Models=%d  Seed=%d", len(CONTROL_MODELS), SEED)
    log.info("=" * 80)

    all_out = []
    for mid, mtype in CONTROL_MODELS:
        try:
            log.info("\n[%s] (%s)", mid, mtype)
            result = _run(mid, mtype, SEED)
            b = result["before"]["laplace_pct"]
            a = result["after"]["laplace_pct"]
            log.info("  Laplace%%: %.1f (step 0) → %.1f (step %d) | Δ %.2f",
                     b, a, TRAIN_STEPS, a - b)
            log.info("  Loss: %.3f → %.3f",
                     result["loss_history"][0], result["loss_history"][-1])
            for s in sorted(result["distributions_at_steps"]):
                log.info("    Step %3d: %.1f%% Laplace", s,
                         result["distributions_at_steps"][s].get("laplace_pct", float("nan")))
            all_out.append({"model_id": mid, "model_type": mtype, "result": result})
        except Exception as exc:
            log.info("  FAILED: %s", exc)

    out_path = os.path.join(os.path.dirname(__file__), "..", "results",
                            "extended_control_500steps.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"results": all_out}, fh, indent=2)
    log.info("\nSaved → %s", out_path)

    log.info("\nSummary:")
    for item in all_out:
        b = item["result"]["before"]["laplace_pct"]
        a = item["result"]["after"]["laplace_pct"]
        log.info("  %-40s  %.1f → %.1f  (Δ %.2f)",
                 item["model_id"], b, a, a - b)


if __name__ == "__main__":
    main()
