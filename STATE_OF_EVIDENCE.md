# State of Evidence

> Last updated: 2026-07-12. Purpose: an honest, disk-verified inventory of what
> the paper currently claims versus what actually exists in `results/`. This file
> is deliberately conservative — it exists so that no claim in `README.md` stands
> unqualified while experiments are still pending. Nothing here is rewritten into
> the paper yet; this is the map that the rewrite (Track C) will follow.
>
> Method note: every line below is sorted into one of three bins —
> **[VERIFIED]** (checked against a file on disk this pass),
> **[OVERCLAIM]** (paper states more than the data currently supports),
> **[BLOCKED]** (needs a rerun we cannot do here). Regenerate the companion
> table any time with `python scripts/reconcile_claims.py`
> (writes `results/reconciliation_table.md`, runs no models).

---

## 1. What is solidly supported (verified on disk)

- **[VERIFIED] The family-level 2-way gradient is real.** Recomputed directly
  from `layerwise_model_comparison.json` (all layers, `ll_laplace` vs
  `ll_gaussian`): encoder-decoder models 100% Laplace-over-Gaussian, RoBERTa-base
  ~92%, BERT-family low (0–17%). Direction matches the paper's Table 1.
- **[VERIFIED] Encoder-decoder attention weights are genuinely peaked.** Median
  GGD shape β: t5-base 0.579, mt5-small 0.537, t5-small 0.586 — sharper than
  Laplace (β=1). This is the strongest, cleanest signal in the whole dataset and
  it is the one with practical (compression) consequences.
- **[VERIFIED] Random-init baseline is Gaussian.** All randomly initialized
  variants sit near β≈2 / Gaussian-dominant, so the peaked structure is acquired
  during training, not at initialization.
- **[VERIFIED] L1 test result exists.** `l1_regularization_results.json` holds the
  control/treatment/delta (12.5% → 100%, +87.5 pp). Single seed — the paper
  already labels it preliminary, which is correct.
- **[VERIFIED] Long-horizon and shuffled controls exist.**
  `extended_control_500steps.json` (gpt2, bert, electra) and
  `shuffled_control_results.json` (gpt2, bert) are present as described.
- **[VERIFIED] Bootstrap null result holds.** `bootstrap_results.json`:
  ρ=0.341, 95% CI [-0.195, 0.794], n=15, 10k replicates. The interval crosses
  zero, so "initialization kurtosis does not predict pretrained Laplace%" stands.

---

## 2. Where the paper currently overclaims — correct before submission

Each item names the exact fix. None require new experiments; they are all
correctable from data already on disk.

- **[OVERCLAIM] "Laplace" as the central noun.** Under the 4-way BIC fit, plain
  Laplace best-fits **zero** of the 153 layers. Winners are GGD or Student-t.
  *Fix:* reframe the finding as "heavy-tailed / sub-Gaussian (generalized-Gaussian
  β<2), with a sub-Laplacian regime (β<1) in encoder-decoder models." Keep
  "Laplace" only as the β=1 reference point, not as the description of the data.
- **[OVERCLAIM] "BERT remains predominantly Gaussian" (Abstract, §4.1, §4.4).**
  Under the 4-way fit, BERT-family layers are predominantly best-fit by
  **Student-t** (heavy-tailed), not Gaussian (bert-base 12/12 Student-t,
  bert-large 13/15). *Fix:* "BERT is the least peaked family, but still
  heavy-tailed (Student-t), not Gaussian." The GPT/BERT *contrast* survives; the
  word "Gaussian" for BERT does not.
- **[OVERCLAIM] Bootstrap numbers in the Abstract/§4.4.** Text says ρ=0.296, CI
  [-0.27, 0.76]; the file says ρ=0.341, CI [-0.195, 0.794]. Conclusion unchanged
  (still null), but the printed numbers are stale. *Fix:* copy the numbers from
  `bootstrap_results.json` verbatim.
- **[OVERCLAIM] "three control experiments totaling 75 training runs" +
  "zero standard deviation across seeds" (§3.4, §4.5).** The short-horizon
  multi-seed control (6 models × 3 seeds) is **not on disk** —
  `extended_control_results.json` contains only a stale `gpt2` block. The
  "zero std across seeds" claim therefore has no supporting data at present.
  *Fix:* either regenerate the short control (Track B) or scope the claim down to
  the long-horizon and shuffled controls that do exist.
- **[OVERCLAIM] Table 1 win-counters presented as canonical.** The paper's Table 1
  comes from the older 2-way, first-8-layers file; the newer all-layers 4-way file
  has its aggregate `laplace_wins`/`gaussian_wins` counters still at 0 (never
  summarized). *Fix:* state explicitly which file/protocol each number comes from,
  and add the 4-way column (see `results/reconciliation_table.md`).
- **[OVERCLAIM] README reproducibility status line.** It marks
  `layerwise_model_comparison.json` as "partial — 1/15 models done"; the file
  actually holds all 15 models, 153 layers. *Fix:* update the status table.

---

## 3. Blocked — needs a GPU rerun (Track B), not a rewrite

Code fixes below were authored in a prior session and are present, uncommitted,
in the working tree. **Epistemic status: verified by reading + syntax check only,
NOT by execution** (this environment has no GPU and the PyTorch network is
blocked). They must actually run and produce output before their results enter
the paper.

- **[BLOCKED] head-level BART.** `head_level_results.json` has 14 models; BART
  failed with `'numpy.ndarray' object has no attribute 'detach'`. Fix applied in
  `ela/analysis.py` (`collect_head_tensors`) — also corrects a latent k/v lookup
  bug (`query.weight` replace vs BART's `q_proj/k_proj/v_proj`). *Confirm:*
  `python scripts/head_level_analysis.py` must run BART without the numpy error.
- **[BLOCKED] short-horizon control.** Fix applied in `scripts/control_short.py`
  + `ela/config.py`: bart/t5 relabeled seq2seq (were `causal` → guaranteed crash),
  and silent failure made loud (previously kept the old file and looked like
  success). *Caveat:* the mislabel does not explain why gpt2 also failed in the
  night run — the true cause is in `runbook_run.log` on the GPU machine
  (unverified hypothesis: OOM from 3 parallel seeds sharing the GPU). *Confirm:*
  the script must write >1 model or exit non-zero.
- **[BLOCKED] LLaMA-3.2-1B.** `modern_llm_results.json` is empty; failure is a
  gated-repo 403. Needs the HuggingFace access form approved, then a rerun.
- **[BLOCKED] Phi-2.** Still commented out in `scripts/modern_llms_ext.py`
  (`LARGE_MODELS`); excluded by the 4 GB VRAM budget.

---

## 4. The honest core that survives all of the above

If every pending item failed tomorrow, this is what the data already supports and
what the paper can claim without a shred of overreach:

> Trained transformer attention weights are heavy-tailed, not Gaussian, and the
> degree of peaking is architecture-dependent: encoder-decoder models (T5/mT5) are
> strongly sub-Laplacian (GGD β≈0.54–0.59), GPT and RoBERTa sit intermediate, and
> BERT is the least peaked but still heavy-tailed. This is acquired in training
> (random init is Gaussian) and is not predicted by initialization kurtosis
> (bootstrap null). Practical compression consequences are a *hypothesis* (§7),
> not yet a measured result.

Everything past that sentence — the phone/efficiency story, the L1 mechanism,
the control-based "architecture-determined" strength — is either single-seed,
partially unsupported, or untested, and should be worded as such until Track B
closes.

---

## 5. Repo hygiene

- Stale `.git/index.lock` (empty) blocks commits; remove on the Mac with
  `rm -f .git/index.lock` before any git operation.
- Uncommitted working-tree edits: `ela/analysis.py`, `ela/config.py`,
  `scripts/control_short.py` (the Track B fixes above). Commit them **separately**
  from Track A's honesty/documentation changes to keep history reviewable.
- `results/layerwise_model_comparison.json` and the uploaded copy are the same
  inode (hardlinked) — never `cp` one onto the other; it truncates both.
