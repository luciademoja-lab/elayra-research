"""
scripts/checkpoint_analysis.py
==============================
Section 8 "training checkpoint analysis" of the paper.

Fine-tunes GPT-2 from a random init (not pretrained — cleaner signal)
for N steps, saving the distributional state at regular checkpoints.
Tests whether the distributional regime is set early (supporting the
gradient-flow hypothesis) or develops gradually (supporting a data-
accumulation account).

Usage
-----
    python scripts/checkpoint_analysis.py
    python scripts/checkpoint_analysis.py --steps 2000 --checkpoint-every 200
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from transformers import AutoConfig, GPT2LMHeadModel, GPT2Tokenizer

from ela.analysis import (
    MAX_LAYERS_PRIMARY,
    collect_attention_tensors,
    summarize_layerwise_fit,
)
from ela.viz import training_trajectory

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

MODEL_ID      = "gpt2"
DEFAULT_STEPS = 500
DEFAULT_CKPT  = 50
BATCH_SIZE    = 8
SEQ_LEN       = 32
LR            = 5e-5
SEED          = 42


def _build_batch(tokenizer, bs: int) -> dict:
    vocab = tokenizer.vocab_size
    ids = torch.randint(0, vocab, (bs, SEQ_LEN))
    return {
        "input_ids":      ids,
        "attention_mask": torch.ones_like(ids),
        "labels":         ids.clone(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps",      type=int, default=DEFAULT_STEPS)
    ap.add_argument("--checkpoint-every", type=int, default=DEFAULT_CKPT)
    ap.add_argument("--output-dir", default=os.path.join("results"))
    args = ap.parse_args()

    torch.manual_seed(SEED)
    import random; random.seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
    config = AutoConfig.from_pretrained(MODEL_ID)
    model = GPT2LMHeadModel(config).to(device)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=LR)

    before = summarize_layerwise_fit(
        collect_attention_tensors(model, max_layers=MAX_LAYERS_PRIMARY)
    )
    snapshots: dict[int, dict] = {0: before}
    loss_history: list[float] = []

    log.info("Training for %d steps (checkpoint every %d)…", args.steps, args.checkpoint_every)
    for step in range(1, args.steps + 1):
        batch = {k: v.to(device) for k, v in _build_batch(tokenizer, BATCH_SIZE).items()}
        out = model(**batch)
        loss = out.loss
        loss_history.append(float(loss))
        loss.backward()
        optim.step()
        optim.zero_grad()

        if step % args.checkpoint_every == 0:
            try:
                snapshots[step] = summarize_layerwise_fit(
                    collect_attention_tensors(model, max_layers=MAX_LAYERS_PRIMARY)
                )
                log.info("  step %4d  loss=%.4f  Laplace=%5.1f%%",
                         step, loss_history[-1],
                         snapshots[step]["laplace_pct"])
            except Exception as exc:
                log.info("  step %4d  checkpoint FAILED: %s", step, exc)

    after = summarize_layerwise_fit(
        collect_attention_tensors(model, max_layers=MAX_LAYERS_PRIMARY)
    )
    snapshots[args.steps] = after

    step_labels  = sorted(snapshots)
    laplace_pcts = [snapshots[s]["laplace_pct"]   for s in step_labels]
    losses_at    = [loss_history[min(s - 1, len(loss_history) - 1)] for s in step_labels]

    png_path = training_trajectory(
        step_labels, laplace_pcts, losses_at,
        title=f"GPT-2 checkpoint analysis  ({args.steps} steps)",
        out_path=os.path.join(args.output_dir, "checkpoint_trajectory.png"),
    )
    log.info("Plot saved → %s", png_path)

    out = {
        "model_id":            MODEL_ID,
        "total_steps":         args.steps,
        "checkpoint_every":    args.checkpoint_every,
        "seed":                SEED,
        "snapshots":           {str(k): v for k, v in snapshots.items()},
        "laplace_pct_trajectory": dict(zip(map(str, step_labels), laplace_pcts)),
    }
    out_path = os.path.join(args.output_dir, "checkpoint_analysis.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    log.info("Saved → %s", out_path)

    # Key finding
    early = snapshots.get(50, snapshots.get(args.steps))
    late  = snapshots[args.steps]
    log.info("\nEarly (step %s): %.1f%% Laplace", list(early.keys())[-1] if isinstance(early, dict) else "?", early.get("laplace_pct", float("nan")) if isinstance(early, dict) else float("nan"))
    log.info("Final (step %d): %.1f%% Laplace", args.steps, late["laplace_pct"])


if __name__ == "__main__":
    main()
