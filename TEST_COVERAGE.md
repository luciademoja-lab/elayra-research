# Test Coverage — structural audit (read-based, not measured)

> Written 2026-07-12. The numbers here are **not** a measured coverage %
> (this analysis was done without torch, so pytest could not run). They are a
> function-by-function map derived by reading `tests/` against the code that
> actually generates results. Measure the real % on a torch machine with:
>
> ```
> python -m pytest tests/ --cov=ela --cov=scripts --cov-report=term-missing
> ```

---

## 1. The headline finding

**The production fit path is untested.** Every test calls
`summarize_layerwise_fit(...)` with the **default** `include_student_t=False` —
i.e. the old **2-way** Laplace/Gaussian branch. The **4-way GGD + Student-t +
BIC** block (`if include_student_t:` in `ela/analysis.py`, ~lines 197–260) is the
branch every real run uses (`analyze_model(..., include_student_t=True)`), and
**no test exercises it.** `ggd_beta`, `ll_ggd`, `ll_student_t`, `aic_winner`,
`bic_winner` are never asserted.

Worse, the current tests actively assert the **old** behaviour:
`better_fit in ("Laplace", "Gaussian")` (test_analysis_core lines 112, 123).
Under the 4-way path `better_fit` is the BIC winner and can be `"GGD"` or
`"Student-t"` — so these assertions encode superseded behaviour.

---

## 2. Coverage map (verified by reading)

### `ela/` — the code the result scripts depend on

| Module / function | Tested? | Notes |
|---|---|---|
| `analysis.collect_attention_tensors` | ✅ good | list/shape/dtype/max_layers/non-zero. Missing: the new bf16 `.float()` path (no bf16 fixture). |
| `analysis.summarize_layerwise_fit` **(2-way)** | ✅ | keys, ranges, win consistency, winner-matches-LL. |
| `analysis.summarize_layerwise_fit` **(4-way BIC)** | ❌ **untested** | the canonical production path. Biggest gap. |
| `analysis.summarize_initialization` | ✅ | stats present, std/range positive. |
| `analysis.analyze_model` | ✅ | keys, model_id, pretrained+random, init_stats. |
| `analysis.collect_head_tensors` | ❌ **imported but no test** | includes Opus's BART fix — unverified by any test. |
| `distributions.*` (ll_all, winner, best_fit, gof_summary, ks_two_sample) | ✅ heavily | **but see §3 — production imports none of this.** |
| `config.*` | ✅ | all dataclass defaults + frozen. |
| `utils.seed_everything`, `build_batch` | ⚠️ partial | only `build_batch(..., "causal")`; masked/seq2seq paths untested. `flush_cuda` untested (trivial). |
| `viz.layer_heatmap`, `bar_comparison` | ✅ smoke | other viz functions untested. |
| `bootstrap.bootstrap_ci` | ✅ | CI keys, NaN guard, short-input raise. |

### `scripts/` — the result-generating entry points

**All of them: 0% direct coverage.** No test imports or invokes any script.
Their logic is only *indirectly* touched where it calls an `ela` function.
Specifically untested script-local logic:

- `run_layerwise_modern._merge_into_checkpoint` (append/replace into the JSON) —
  *I verified this logic in isolation, but it is not in the suite.*
- `run_layerwise_incremental` checkpoint/resume/`--reset` logic.
- `mlp_embedding_analysis`: `collect_mlp_tensors`, `collect_embedding_tensors`
  (tied-weight dedup), `param_census`, `_median_beta`, `_bic_counts`.
- `reconcile_claims`: `two_way_table`, `four_way_table`, `write_markdown`.
- every `main()` (arg parsing, output writing).

---

## 3. Obsolete / low-value tests (testing code the system doesn't use)

- **All `ela.distributions` tests** — `TestDistributions` (test_analysis_core,
  ~lines 184–204) and `TestDistributionsGOF` (test_ela_extended, ~lines 105–149).
  Verified: **no production script imports `ela.distributions`** — it is imported
  only by these tests. They validate a parallel 3-way fit + KS API that nothing
  in the pipeline runs. They are not *wrong*, but they spend coverage on
  dead-in-production code while the real 4-way fit goes untested. When the refactor
  consolidates fitting (REPO_MAP §6.3), these should be repointed at the canonical
  `summarize_layerwise_fit`, not deleted blindly (KS goodness-of-fit is worth
  keeping — but as a test of the code the paper actually uses).
- **The 2-way `better_fit` assertions** (test_analysis_core `test_each_layer_has_ll`,
  `test_winner_matches_lls`) — assert `better_fit ∈ {Laplace, Gaussian}`, which is
  the pre-4-way behaviour. Obsolete as written; must be updated when the 4-way
  path is added to the suite.

Also relevant (not tests, but "unused code" the console still points at):
`pyproject.toml [project.scripts]` still registers `ela-mlp = mlp_analysis`
(the old 2-way) and `ela-modern = modern_llms_ext` (superseded) — the retire
candidates from REPO_MAP §5.

---

## 4. Reaching 95–98% — honestly

A blanket 95–98% over `ela` **and** `scripts` is the wrong target: the scripts'
`main()` functions load multi-GB models and write files — that's integration
territory, expensive and low-value to unit-test, and it will drag the number down
no matter what. Two honest options:

**Option A (recommended): scope coverage to the logic, hit 95–98% there.**
Measure `--cov=ela` plus the *pure helpers* of scripts, and mark model-loading
mains as integration. Concretely, to close the gaps above:

1. **4-way BIC path** — add synthetic tests: feed `summarize_layerwise_fit(..., include_student_t=True)` samples drawn from GGD β=0.5 / Laplace / Gaussian / Student-t and assert `bic_winner` and recovered `ggd_beta` (the exact recovery test already validated by hand). *Highest value — closes the biggest gap.*
2. **`collect_head_tensors`** — one test on GPT-2 (fused `c_attn`) and one on a separate-projection model; asserts per-head shapes. Locks in Opus's BART fix.
3. **New pure logic** — unit-test `_merge_into_checkpoint`, `param_census` (tied-weight dedup), `collect_mlp/embedding_tensors`, `_median_beta`, `reconcile_claims` table builders. These are fast, need no GPU, and are where regressions will bite.
4. **Fix the obsolete assertions** so they accept 4-way winners.
5. Add `# pragma: no cover` to the `if __name__ == "__main__":` blocks and any
   GPU-only branch, so the denominator is meaningful.

**Option B: full end-to-end coverage** — smoke-run each script on the smallest
model. Achieves high line coverage but is slow and mostly re-tests transformers,
not your science. Not recommended as the primary path.

**Precondition:** the REPO_MAP refactor makes 95–98% *achievable and meaningful* —
moving logic out of the script `main()`s into importable `ela` functions is what
turns "untestable CLI glue" into "unit-testable functions." Do the coverage push
**after** the unify-runs refactor, not before, or you'll write tests against files
that are about to move.

---

## 5. One-line summary

Today's suite covers the `ela` primitives' **2-way** path and the (unused)
`distributions` module well, but **does not touch the 4-way BIC path that every
real run uses, `collect_head_tensors`, or any script**. The fastest route to a
meaningful 95–98% is: add 4-way synthetic tests, unit-test the new pure helpers,
retire/repoint the `distributions` tests — after the refactor moves logic into
`ela`.
