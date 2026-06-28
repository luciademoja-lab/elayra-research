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

import concurrent.futures
import gc
import json
import logging
import os
import sys
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

from ela.analysis import (
    MAX_LAYERS_PRIMARY,
    collect_attention_tensors,
    summarize_layerwise_fit,
)
from ela.config import ControlShortConfig
from ela.utils import build_batch, flush_cuda, seed_everything

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


cfg = ControlShortConfig()


def _run_seed(model_id: str, model_type: str, seed: int) -> dict:
    seed_everything(seed)

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

    losses: List[float] = []
    for _ in range(cfg.train_steps):
        batch = {k: v.to(device) for k, v in build_batch(
            tokenizer, cfg.batch_size, cfg.seq_len, model_type).items()}
        out = model(**batch)
        loss = out.loss
        losses.append(float(loss))
        loss.backward()
        optim.step()
        optim.zero_grad()

    after = summarize_layerwise_fit(
        collect_attention_tensors(model, max_layers=MAX_LAYERS_PRIMARY)
    )

    del model, optim, batch, out
    gc.collect()
    flush_cuda()

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
             len(cfg.model_ids), cfg.num_seeds, cfg.train_steps, cfg.batch_size)
    log.info("=" * 80)

    all_out = []
    for mid, mtype in tqdm(cfg.model_ids, desc="Models", unit="model"):
        try:
            log.info("\n[%s] (%s)", mid, mtype)
            # Thread-based parallelism keeps CUDA context in the main process
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=cfg.num_seeds) as pool:
                futures = {
                    pool.submit(_run_seed, mid, mtype, s): s
                    for s in range(cfg.num_seeds)
                }
                seeds = []
                for fut in tqdm(concurrent.futures.as_completed(futures),
                                total=len(futures),
                                desc=f"  {mid} seeds",
                                leave=False):
                    seeds.append(fut.result())

            agg = _aggregate(seeds)
            all_out.append({"model_id": mid, "model_type": mtype, "aggregated": agg})
            for sr in seeds:
                log.info("  seed %d: %.1f%% → %.1f%%  (loss=%.4f)",
                         sr["seed"], sr["before"]["laplace_pct"],
                         sr["after"]["laplace_pct"], sr["final_loss"])
            log.info("  mean: %.1f (±%.1f) → %.1f (±%.1f)  Δ %.2f",
                     agg["before_mean"], agg["before_std"],
                     agg["after_mean"],  agg["after_std"], agg["change_mean"])
        except Exception as exc:
            log.info("  FAILED: %s", exc)

    flush_cuda()

    out_path = os.path.join(os.path.dirname(__file__), "..", "results",
                            "extended_control_results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"results": all_out}, fh, indent=2)
    log.info("\nSaved → %s", out_path)


if __name__ == "__main__":
    main()
