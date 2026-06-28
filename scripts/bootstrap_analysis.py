"""
scripts/bootstrap_analysis.py
=============================
Computes the bootstrap 95 % CI for the Spearman ρ between initialisation
kurtosis and pretrained Laplace% using existing results JSON.

Current result (10,000 replicates): ρ = 0.296, CI95 [-0.27, 0.76].

Usage
-----
    python scripts/bootstrap_analysis.py
    python scripts/bootstrap_analysis.py --input results/broader_analysis_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ela.bootstrap import bootstrap_ci
from ela.config import PipelineConfig
from ela.utils import flush_cuda

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

_DEFAULT_INPUT = os.path.join("results", "broader_analysis_results.json")


def main() -> None:
    cfg = PipelineConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=_DEFAULT_INPUT, help="Path to broader_analysis_results.json")
    ap.add_argument("--n", type=int, default=cfg.bootstrap_n, help="Bootstrap replicates")
    ap.add_argument("--seed", type=int, default=cfg.bootstrap_seed)
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as fh:
        data = json.load(fh)

    results = data["results"]
    kurt_vals = [r["init_stats"]["kurtosis"] for r in results]
    lap_pcts  = [r["pretrained"]["laplace_pct"]  for r in results]

    ci = bootstrap_ci(kurt_vals, lap_pcts, n=args.n, seed=args.seed)

    log.info("=" * 60)
    log.info("BOOTSTRAP ANALYSIS")
    log.info("=" * 60)
    log.info("Observations   : %d", len(results))
    log.info("Spearman ρ     : %.4f", ci["stat"])
    log.info("95%% CI         : [%.4f, %.4f]", ci["ci_low"], ci["ci_high"])
    log.info("Replicates     : %d", ci["n"])

    out = {
        "n_observations": len(results),
        "spearman_rho":   round(ci["stat"], 6),
        "ci_95_low":      round(ci["ci_low"], 6),
        "ci_95_high":     round(ci["ci_high"], 6),
        "n_bootstrap":    int(ci["n"]),
        "seed":           args.seed,
    }
    out_path = os.path.join(os.path.dirname(args.input), "bootstrap_results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    log.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
