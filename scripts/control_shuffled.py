"""
scripts/control_shuffled.py
===========================
Replacement for shuffled_control_experiment.py.

Minimal training-signal control: 10-step fine-tune on GPT-2 and BERT-base
with fully randomised labels.  Confirms that even the most minimal
gradient signal does not disrupt the architecture-determined
distributional regime (Section 4.5).

Usage
-----
    python scripts/control_shuffled.py
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
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

from ela.analysis import (
    MAX_LAYERS_PRIMARY,
    collect_attention_tensors,
    summarize_layerwise_fit,
)
from ela.config import ShuffleControlConfig
from ela.utils import build_batch, flush_cuda, seed_everything

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


cfg = ShuffleControlConfig()


def _run(model_id: str, model_type: str) -> dict:
    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    ctor = (AutoModelForCausalLM if model_type == "causal"
            else AutoModelForMaskedLM)
    model = ctor.from_pretrained(model_id).to(device)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    before = summarize_layerwise_fit(
        collect_attention_tensors(model, max_layers=MAX_LAYERS_PRIMARY)
    )

    for _ in tqdm(range(cfg.train_steps), desc=f"  {model_id}", leave=False):
        batch = {k: v.to(device) for k, v in build_batch(
            tokenizer, cfg.batch_size, cfg.seq_len, model_type).items()}
        out = model(**batch)
        loss = out.loss
        loss.backward()
        optim.step()
        optim.zero_grad()

        del batch

    after = summarize_layerwise_fit(
        collect_attention_tensors(model, max_layers=MAX_LAYERS_PRIMARY)
    )

    del model, optim
    gc.collect()
    flush_cuda()

    return {"model_id": model_id, "model_type": model_type,
            "before": before, "after": after}


def main() -> None:
    log.info("=" * 80)
    log.info("SHUFFLED-LABEL CONTROL  (ela-backed, %d steps)", cfg.train_steps)
    log.info("=" * 80)

    results = []
    for mid, mtype in tqdm(cfg.model_ids, desc="Models", unit="model"):
        try:
            log.info("[%s] (%s)", mid, mtype)
            r = _run(mid, mtype)
            results.append(r)
            log.info("  before: %.1f%% Laplace", r["before"]["laplace_pct"])
            log.info("  after:  %.1f%% Laplace", r["after"]["laplace_pct"])
        except Exception as exc:
            log.info("  FAILED: %s", exc)

    out_path = os.path.join(os.path.dirname(__file__), "..", "results",
                            "shuffled_control_results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"results": results}, fh, indent=2)
    log.info("\nSaved → %s", out_path)


if __name__ == "__main__":
    main()
