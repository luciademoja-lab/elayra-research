"""
scripts/run_layerwise_incremental.py
======================================
Process one model at a time for the layerwise analysis.

Usage:
    python scripts/run_layerwise_incremental.py              # process next pending model
    python scripts/run_layerwise_incremental.py gpt2         # process a specific model
    python scripts/run_layerwise_incremental.py --list       # show pending models
    python scripts/run_layerwise_incremental.py --reset      # clear checkpoint
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ela.analysis import MODEL_IDS, MAX_LAYERS_DEPTH, analyze_model
from ela.utils import flush_cuda

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "layerwise_model_comparison.json"
)


def _load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"results": [], "failures": [], "pending": list(MODEL_IDS)}


def _save_checkpoint(state: dict) -> None:
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


def _next_model(state: dict, requested: str | None = None) -> str | None:
    if requested:
        if requested in state["pending"]:
            return requested
        if requested in [r["model_id"] for r in state["results"]]:
            log.info("Model %s already processed.", requested)
            return None
        log.warning("Model %s not in pending list.", requested)
        return None
    if state["pending"]:
        return state["pending"][0]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Layerwise incremental processor")
    parser.add_argument("model", nargs="?", help="Specific model_id to process")
    parser.add_argument("--list", action="store_true", help="Show pending models")
    parser.add_argument("--reset", action="store_true", help="Clear checkpoint and restart")
    args = parser.parse_args()

    if args.reset:
        state = {"results": [], "failures": [], "pending": list(MODEL_IDS)}
        _save_checkpoint(state)
        log.info("Checkpoint reset. All %d models pending.", len(MODEL_IDS))
        return

    state = _load_checkpoint()

    if args.list:
        done = {r["model_id"] for r in state["results"]}
        failed = {f["model_id"] for f in state["failures"]}
        pending = [m for m in MODEL_IDS if m not in done and m not in failed]
        log.info("Processed: %d", len(done))
        log.info("Failed: %d", len(failed))
        log.info("Pending: %d", len(pending))
        for mid in pending:
            log.info("  - %s", mid)
        return

    mid = _next_model(state, args.model)
    if mid is None:
        log.info("Nothing to process.")
        return

    log.info("=" * 80)
    log.info("LAYERWISE INCREMENTAL  depth=%d  remaining=%d", MAX_LAYERS_DEPTH, len(state["pending"]))
    log.info("=" * 80)
    log.info("Processing: %s", mid)

    try:
        r = analyze_model(mid, max_layers=MAX_LAYERS_DEPTH, include_student_t=True)
        state["results"].append(r)
        state["pending"].remove(mid)
        log.info("  pretrained: %d layers  %.1f%% Laplace",
                 r["pretrained"]["num_layers"], r["pretrained"]["laplace_pct"])
        st = r["pretrained"].get("student_t_wins", "n/a")
        log.info("  Student-t wins: %s", st)
    except Exception as exc:
        state["failures"].append({"model_id": mid, "error": str(exc)})
        state["pending"].remove(mid)
        log.info("  FAILED: %s", exc)

    flush_cuda()
    _save_checkpoint(state)

    remaining = len(state["pending"])
    done = len(state["results"])
    failed = len(state["failures"])
    log.info("\nCheckpoint saved: %d done, %d failed, %d pending", done, failed, remaining)


if __name__ == "__main__":
    main()
