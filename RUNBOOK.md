# RUNBOOK — Full analysis rerun (zero-footprint Windows setup)

> **If you are a Claude / Cowork session on a new machine:** this file is your
> briefing. The project memory (CLAUDE.md) lives on Lucia's Mac and is NOT
> available here — everything you need is in this file and in README.md.
>
> **Goal of this run:** regenerate ALL results with the current pipeline
> (deterministic 500k subsampling + 4-way BIC classification:
> Laplace / Gaussian / Student-t / Generalized Gaussian). The numbers in the
> README's Results sections are stale and will be rewritten from this run's
> output — do NOT trust or preserve them.

## 0. Ground rules

- **Touch nothing outside the repo folder.** The setup below keeps the venv,
  the Python interpreter, and the HuggingFace model cache all inside the repo.
- **Check the GPU is free first** (`nvidia-smi`): close any other Python
  compute processes before starting.
- Fits run on **CPU** (scipy); the GPU only accelerates the training-based
  experiments (controls, L1, checkpoint).

## 1. One-time setup (PowerShell, no admin rights)

```powershell
# 1. Install uv (single user-level binary; skip if `uv --version` works)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. From the repo root: create an in-repo venv with Python 3.12.
#    (uv downloads its own CPython — system Python 3.13/3.14 are NOT supported)
cd <repo>
uv venv --python 3.12
.venv\Scripts\Activate.ps1

# 3. torch FIRST, matched to the GPU.
#    RTX 50-series (Blackwell) REQUIRES cu128 — cu121 wheels will not run:
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
#    (CPU-only fallback: uv pip install torch)

# 4. Remaining dependencies + the package itself (editable, scripts import `ela`)
uv pip install -r requirements-lock.txt
uv pip install -e .

# 5. Keep the model cache INSIDE the repo (~25 GB; set in every new shell)
$env:HF_HOME = "$PWD\hf_cache"

# 6. Optional, only for gated models (LLaMA, Gemma):
# $env:HF_TOKEN = "hf_..."
```

## 2. Sanity check — run the tests FIRST

```powershell
python -m pytest tests/ -v        # all 47 tests must pass before any run
```

## 3. Environment knobs

| Variable | Default | Meaning |
|---|---|---|
| `ELA_SUBSAMPLE` | `500000` | Per-layer sample size for fitting. `0` = full tensor (SLOW: hours/layer for Student-t). |
| `ELA_SUBSAMPLE_SEED` | `12345` | Per-layer seed (`seed + layer_index`); identical runs on any machine. |

## 4. Run order

```powershell
# (a) Layerwise 4-way, 15 models — THE core result. RESET FIRST: the current
#     checkpoint holds one model (gpt2) computed under the OLD protocol
#     (full-tensor, raw-LL 3-way) and must not be mixed with new entries.
python scripts\run_layerwise_incremental.py --reset
python scripts\run_layerwise.py                      # ~30–90 min total
#     (alternative, resumable one-model-at-a-time:
#      loop `python scripts\run_layerwise_incremental.py` until "Nothing to process";
#      `--list` shows pending)

# (b) Primary 8-layer protocol + random-init comparison
python scripts\run_pipeline.py

# (c) The two results cited in the paper but never saved to disk
python scripts\l1_regularization_test.py
python scripts\bootstrap_analysis.py

# (d) Randomized-label controls (GPU recommended)
python scripts\control_short.py
python scripts\control_long.py
python scripts\control_shuffled.py

# (e) Secondary analyses
python scripts\init_analysis.py
python scripts\head_level_analysis.py               # BART previously failed: numpy
                                                    # .detach() error — if it fails
                                                    # again, log it, don't block
python scripts\mlp_analysis.py
python scripts\checkpoint_analysis.py

# (f) Extended models (needs HF_TOKEN for LLaMA; Phi-2 works without)
$env:INCLUDE_OPTIONAL_MODELS = "1"
python scripts\modern_llms_ext.py

# (g) Subsample stability check (for the paper's appendix):
#     full-tensor runs on the three SMALL models only + one alternate seed.
python scripts\run_layerwise_incremental.py --reset  # after saving (a)!  see note
$env:ELA_SUBSAMPLE = "0"
python scripts\run_layerwise_incremental.py distilbert-base-uncased
python scripts\run_layerwise_incremental.py t5-small
python scripts\run_layerwise_incremental.py google/electra-small-discriminator
$env:ELA_SUBSAMPLE = "500000"; $env:ELA_SUBSAMPLE_SEED = "99"
python scripts\run_layerwise_incremental.py distilbert-base-uncased
# NOTE: (g) overwrites results/layerwise_model_comparison.json — copy the (a)
# output to a safe name (e.g. layerwise_main.json) BEFORE starting (g), and
# rename (g)'s outputs (e.g. layerwise_fulltensor.json / layerwise_seed99.json).

# (h) Figures (needs the JSONs above)
python scripts\generate_figures.py
```

## 5. What the new output fields mean

Each layer entry now records: `n_total` / `n_used` (subsampling provenance),
`ll_laplace` / `ll_gaussian` / `ll_student_t` / `ll_ggd`, `student_t_df`
(tail-heaviness dial: ~1–3 = very heavy, →∞ = Gaussian), `ggd_beta`
(1 = Laplace, 2 = Gaussian), `rawll_winner` / `aic_winner` / `bic_winner`,
and `better_fit` (**= BIC winner**; raw LL is biased toward the 3-parameter
families — verified on synthetic data — so BIC is the primary criterion).

## 6. After the run

1. Verify: every JSON in `results/` regenerated, `results/figures/` populated.
2. `git add results/ && git commit -m "results: full 4-way BIC rerun (500k subsample)"`.
3. Push. The paper rewrite (abstract, Table 1, §4, Conclusion) happens ONLY
   after these numbers are in.

## Troubleshooting

- `no kernel image is available` → torch is cu121 on a Blackwell GPU; reinstall cu128 (step 3).
- transformers refuses `.bin` checkpoints (ELECTRA, mT5, opt) → torch must be ≥ 2.6; cu128 wheel satisfies this.
- `ModuleNotFoundError: ela` → `uv pip install -e .` not run, or venv not activated.
- Python 3.13/3.14 wheel errors → the venv must be 3.12 (step 2).
- Slow t-fits (minutes/layer) → `ELA_SUBSAMPLE` accidentally `0`.
