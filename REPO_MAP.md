# Repo Map — how the pieces fit (and where the cruft is)

> Written 2026-07-12 as the blueprint for an eventual refactor. **This document
> changes no code.** It exists so we both stop guessing which file does what, and
> so the future cleanup is executed from a plan rather than freehand.
>
> Everything below is **verified from disk** (script outputs, imports, and
> reads were grepped, not remembered) unless marked *(inferred)*.

---

## 1. What the project is

An empirical study of transformer weight-distribution regimes: fit Laplace /
Gaussian / Student-t / generalized-Gaussian (GGD) to attention (and now MLP +
embedding) weights across model families, under BIC. The pivot is toward
distribution-aware quantization for on-device models.

Core library: `ela/`. Entry points: `scripts/`. Outputs: `results/` (+ `figures/`).

---

## 2. Data flow — script → output → who consumes it

| Stage | Script | Writes | Consumed by |
|---|---|---|---|
| Primary 8-layer (2-way, Table 1) | `run_pipeline.py` | `broader_analysis_results.json` | `generate_figures.py`, `bootstrap_analysis.py`, paper Table 1 |
| Layerwise 15-layer **(4-way, canonical)** | `run_layerwise.py` **or** `run_layerwise_incremental.py` **or** `run_layerwise_modern.py` | `layerwise_model_comparison.json` | `generate_figures.py`, `reconcile_claims.py` |
| Init statistics | `init_analysis.py` | `expanded_model_init_results.json` | `generate_figures.py` |
| Bootstrap CI | `bootstrap_analysis.py` (reads broader) | `bootstrap_results.json` | `generate_figures.py`, paper §4.4 |
| L1 test | `l1_regularization_test.py` | `l1_regularization_results.json` | `generate_figures.py`, paper §4.6 |
| Controls | `control_short.py` / `control_long.py` / `control_shuffled.py` | `extended_control_results.json` / `extended_control_500steps.json` / `shuffled_control_results.json` | `generate_figures.py`, paper §4.5 |
| Head-level | `head_level_analysis.py` | `head_level_results.json` (+ heatmap) | `generate_figures.py`, paper §5.2 |
| Checkpoint | `checkpoint_analysis.py` | `checkpoint_analysis.json` (+ png) | paper (checkpoint trajectory) |
| MLP (old, 2-way) | `mlp_analysis.py` | `mlp_analysis_results.json` | **nobody reads it** (see §5) |
| MLP + embedding (new, 4-way) | `mlp_embedding_analysis.py` | `mlp_embedding_4way.json` | (new; feeds compression argument) |
| Modern LLMs (old) | `modern_llms_ext.py` | `modern_llm_results.json` | **nobody reads it** (see §5) |
| Figures | `generate_figures.py` | `figures/fig1..19.png` | paper |
| Honesty tooling (new) | `reconcile_claims.py` | `reconciliation_table.md` | STATE_OF_EVIDENCE |

**Archives (written by hand, not by any script — verified orphan of producers):**
`layerwise_main.json`, `layerwise_fulltensor.json`, `layerwise_seed99.json`.
These are RUNBOOK step (g) products (canonical main run + subsample-stability
appendix). **Not stale — protect (see §4).**

---

## 3. `ela/` core modules

| Module | Holds | Imported by |
|---|---|---|
| `analysis.py` | model loading, weight collectors, **`summarize_layerwise_fit` (the real 4-way GGD+BIC fit)**, init stats, head collectors, `MODEL_IDS` | almost every script |
| `distributions.py` | a **separate, older 3-way** (Laplace/Gaussian/Student-t) fit + KS helpers | **tests only** — no script imports it (see §5) |
| `config.py` | dataclass configs for the training-based experiments | control/l1/checkpoint/bootstrap/pipeline |
| `utils.py` | `build_batch`, `flush_cuda`, `seed_everything` | widely |
| `viz.py` | shared plotting | `generate_figures`, `checkpoint`, `head_level` |
| `bootstrap.py` | `bootstrap_ci` | `bootstrap_analysis`, `run_pipeline` |

---

## 4. "Looks stale but is NOT" — do not delete

- `layerwise_main.json` — canonical main 4-way run (the night run). The uploaded
  `layerwise_model_comparison.json` is hardlinked to a working copy; `_main` is
  the safe archive. **Keep.**
- `layerwise_fulltensor.json`, `layerwise_seed99.json` — subsample-stability
  appendix (ELA_SUBSAMPLE=0 and alternate seed). **Keep.**
- `broader_analysis_results.json` — still the source of the paper's Table 1
  (2-way, 8-layer). Not superseded until the paper migrates to the 4-way file. **Keep.**
- `distributions.py` — parallel 3-way API, but **the test suite imports it**
  (`test_ela_extended`, `test_analysis_core`). Deleting it breaks tests. Consolidate,
  don't delete (see §6).

---

## 5. Genuinely retire-able — but verify the gate first

Each of these looks superseded. Before removing any, run the stated check.

- `mlp_analysis.py` + `mlp_analysis_results.json` — old 2-way MLP fit, **read by
  nobody** (verified: `generate_figures` does not load it). Superseded by
  `mlp_embedding_analysis.py`. *Gate:* confirm the paper's MLP section will cite
  the 4-way output, then retire the script and repoint the `mlp` Makefile target.
- `modern_llms_ext.py` + `modern_llm_results.json` — old modern-LLM path that
  only ever recorded the LLaMA 403. **Read by nobody.** Superseded by
  `run_layerwise_modern.py` (which merges into the layerwise file). *Gate:* confirm
  nothing in `LARGE_MODELS` there is still wanted, then retire.

---

## 6. Overlap / duplication hotspots — the actual refactor targets

Ranked by payoff. This is the "unify the runs" you asked about.

1. **Three scripts, one output.** `run_layerwise.py` (batch-15),
   `run_layerwise_incremental.py` (resumable one-at-a-time), and
   `run_layerwise_modern.py` (LLaMA+Phi merge) **all write
   `layerwise_model_comparison.json`**. → Collapse into ONE runner with flags:
   `--models all|<id>...|modern`, `--resume/--reset`, `--max-layers`. One code
   path, one checkpoint logic, no drift.
2. **Two MLP scripts.** `mlp_analysis.py` (2-way) vs `mlp_embedding_analysis.py`
   (4-way + embeddings + census). → Keep the 4-way; retire the 2-way (§5).
3. **Two fitting implementations.** `analysis.summarize_layerwise_fit` (4-way,
   canonical, used in production) vs `distributions.py` (3-way, tests only). →
   Make `distributions.py` a thin set of atoms that `analysis` also uses, so there
   is a single source of truth for "how a layer is fit," and update the tests to it.
4. **`run_pipeline.py` vs `run_layerwise.py`.** Both call `analyze_model`; one caps
   at 8 layers/2-way framing, the other 15/4-way. → Same unified runner with a
   `--depth` and `--protocol` flag; `broader_analysis` becomes just "the 8-layer
   preset."
5. **Model registries scattered.** `MODEL_IDS` / `OPTIONAL_MODEL_IDS` /
   `CAPABLE_MODEL_IDS` in `analysis.py`, `MODERN_MODEL_IDS` in the modern script,
   per-experiment lists in `config.py`. → One registry with tags (family, gated,
   size-tier) that every runner reads.

---

## 7. Small inconsistencies (cheap to fix, low risk)

- `generate_figures.py` has **no Makefile target** (RUNBOOK calls it directly). Add `make figures`.
- `reconcile_claims.py` and `run_layerwise_modern.py` are only partly wired. Fine, but add Makefile targets for discoverability.
- Makefile `PY := .venv/Scripts/python.exe` is **Windows-only**; the Mac venv is `.venv/bin/python`. A cross-OS `PY` (detect on run) would let `make` work on both.

---

## 8. Proposed target structure (the blueprint)

```
ela/
  registry.py        # one tagged model registry (replaces the 3 scattered lists)
  fitting.py         # the single 4-way GGD+BIC fit (merges analysis+distributions fit logic)
  collect.py         # weight collectors (attention / mlp / embedding / head) in one place
  train.py           # build_batch / seed / the control-training loop (shared by controls+L1)
  viz.py, utils.py, config.py   # as today

scripts/
  run.py             # ONE entry point:  run.py analyze --models ... --depth ... --resume
                     #                    run.py controls | init | bootstrap | l1 | figures
  (thin wrappers kept only if a stable CLI name is needed)

results/             # unchanged; add a short results/README.md indexing each file
```

Net effect: the three layerwise runners, the two MLP scripts, and the two fit
implementations collapse to one each. A reader (human or model) finds "the code
that fits a layer" in exactly one place.

---

## 9. Refactor safety protocol (when we actually do it)

1. **Not now, not on `broader-analysis`.** Land the in-flight runs first, then
   branch: `git switch -c refactor/unify-runs`.
2. **One move per commit**, each guarded by `python -m pytest tests/ -v` on a
   torch-capable machine (the 47 tests are the safety net — a refactor that can't
   run them is unverified).
3. **Deletions last**, and only after `grep -rn "<name>"` across `scripts/ ela/
   tests/ *.md Makefile *.ps1` shows zero live references, and the data is already
   in git history.
4. **Keep outputs byte-stable** where possible: if a JSON schema must change,
   regenerate `reconciliation_table.md` and diff, so no silent number drift.
5. Update `RUNBOOK.md`, `Makefile`, `run_extend.ps1`, and this map in the **same**
   commit as any rename, so the one-command flow never points at a moved file.
