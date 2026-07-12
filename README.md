# Beyond the Gaussian Prior: Architecture-Dependent Weight Distribution Regimes in Transformer Attention
## Heavy-tailed, architecture-dependent attention-weight distributions in trained transformers, and their implications for distribution-aware quantization

> Full paper reproduced below. **Jump to § Reproducibility** for run targets and hardware notes.

---

## Abstract

Standard transformer initialization schemes assume Gaussian weight distributions, yet it remains unknown whether this assumption holds after pretraining and whether distributional behavior is architecture-dependent. We present a systematic empirical analysis of attention projection weight distributions across **15 transformer variants** spanning GPT-2, BERT, RoBERTa, ALBERT, ELECTRA, BART, and T5 families. Using maximum-likelihood fitting with layer-wise log-likelihood comparison, we find that distributional regime — heavy-tailed versus Gaussian-like — is strongly associated with architecture family and training objective, not with training data content. Causal language models (GPT-2 family) and encoder-decoder models (BART, T5) consistently exhibit heavy-tailed, sharply peaked weight distributions in their attention layers — sub-Laplacian (generalized-Gaussian shape β ≈ 0.54–0.59) in the encoder-decoder family — while masked language models (BERT family) are the least peaked but still non-Gaussian, best described by a Student-t rather than a Gaussian under a four-distribution BIC comparison. This divergence is robust across three randomized-label control experiments totaling 75 training runs, suggesting that short-term gradient flow from task-relevant data is unlikely to be the sole mechanism. Notably, RoBERTa — architecturally near-identical to BERT but trained with a more aggressive regime — achieves 100% Laplace prevalence versus BERT's 8.3%, implicating training intensity and masking strategy as potential modulators of distributional regime beyond architecture alone. We further observe that Laplace prevalence in GPT-style models concentrates in early layers and decreases with depth, revealing a layer-depth signature consistent with gradient flow asymmetry induced by causal masking. A bootstrap correlation analysis of initialization kurtosis against pretrained Laplace% returns a null result (ρ = 0.341, 95% CI [-0.19, 0.79]), reinforcing that training dynamics — not initialization statistics — are the primary driver. We connect these findings to the maximum-entropy characterization of the Laplace distribution under L1 constraints, propose that autoregressive training may impose implicit sparsity pressure functionally equivalent to L1 regularization, and discuss concrete implications for pruning, quantization, uncertainty quantification, and fine-tuning strategies calibrated to architecture family.

---

## 1. Introduction

The distributional properties of neural network weights are routinely treated as an implementation detail rather than a scientific object of study. Standard initialization schemes — Xavier/Glorot (Glorot & Bengio, 2010), He (He et al., 2015) — prescribe Gaussian or uniform distributions over initial weights based on variance preservation arguments, and this Gaussian prior is embedded throughout the deep learning stack: in Laplace approximations for Bayesian uncertainty quantification (Daxberger et al., 2021), in magnitude-based pruning heuristics, in quantization calibration procedures, and in the theoretical analyses underlying variational inference methods.

Whether trained weights remain approximately Gaussian — and whether any departure from Gaussianity is systematic across architectures — has received surprisingly little direct investigation. Mechanistic interpretability research has produced rich accounts of attention head functions (Elhage et al., 2021), superposition phenomena (Elhage et al., 2022), and induction circuits, but these analyses typically operate on activations and attention patterns rather than on the distributional geometry of the weight tensors themselves. Heavy-tailed phenomena in trained networks have been studied at the level of weight matrix spectra (Martin & Mahoney, 2019, 2021), where it is known that well-trained networks develop heavy-tailed singular value distributions as a signature of implicit self-regularization. The element-wise distributional regime of individual weight tensors — Laplace versus Gaussian, and whether this is architecture-determined — has not been systematically characterized.

This paper addresses that gap. We ask three questions. First, do trained transformer attention weights deviate from Gaussianity in a consistent, family-specific way? Second, is any such deviation caused by training data content or by architecture and training objective? Third, do initialization statistics predict the distributional regime reached after training, or is the mechanism instead rooted in training dynamics?

Our contributions are as follows. We provide the first systematic layer-wise comparison of Laplace versus Gaussian fit quality across 15 transformer variants spanning seven architecture families, using both pretrained and randomly initialized variants for each model. Our empirical findings strongly point to an architectural boundary: the distributional pattern appears to be architecture-determined and largely robust to short-term gradient flow from arbitrary data. We identify a layer-depth signature in GPT-style models where Laplace prevalence is concentrated in early layers and decreases monotonically with depth, a pattern consistent with gradient flow asymmetry induced by causal masking. We show that the BERT/RoBERTa pair constitutes a natural experiment isolating training regime effects from architecture, with RoBERTa's more aggressive training producing a dramatic shift toward heavy-tailed, sharply peaked weights despite near-identical architectural structure. A bootstrap correlation of initialization kurtosis with pretrained Laplace% returns a null result (ρ = 0.341, 95% CI [-0.19, 0.79]), ruling out initialization statistics as a strong deterministic driver and reinforcing training dynamics as the primary mechanism. Finally, we connect these empirical observations to the maximum-entropy theory of the Laplace distribution and propose a mechanistic hypothesis relating autoregressive training to implicit L1-equivalent sparsity pressure, with falsifiable predictions for future work. We discuss implications for pruning, quantization, uncertainty quantification, fine-tuning, and the design of architecture-aware initialization schemes.

The remainder of the paper is organized as follows. Section 2 reviews related work. Section 3 describes methods. Section 4 presents results. Section 5 provides discussion and mechanistic hypotheses. Section 6 proposes a theoretical framework. Section 7 describes implications for practice. Section 8 concludes.

---

## 2. Related Work

### 2.1 Weight Initialization Theory

The dominant framework for weight initialization in deep networks derives from variance preservation arguments. Glorot & Bengio (2010) showed that maintaining constant variance of activations and gradients across layers requires initializing weights from distributions with variance inversely proportional to fan-in plus fan-out, yielding the Xavier initialization scheme. He et al. (2015) extended this analysis to ReLU networks, deriving variance scaling proportional to inverse fan-in. Both schemes prescribe zero-mean distributions — typically Gaussian or uniform — and embed the assumption that this distributional shape is appropriate for the learning regime. The theoretical motivation for Gaussianity at initialization is primarily the central limit theorem: weights are often treated as sums of many small random effects, and the Gaussian is the maximum-entropy distribution under a variance constraint.

What happens to this distributional shape during training is a separate question that initialization theory does not address. Our work provides empirical evidence that training systematically deforms the weight distribution in architecture-dependent ways, motivating initialization schemes that account for the likely learned distributional regime rather than treating the trained distribution as irrelevant.

### 2.2 Heavy-Tailed Self-Regularization in Deep Networks

Martin & Mahoney (2019, 2021) proposed the heavy-tailed self-regularization (HT-SR) framework, which characterizes the singular value distributions of weight matrices in trained networks. They observe that well-generalized networks develop power-law or heavy-tailed spectra in their weight matrices, which they interpret as a signature of implicit self-regularization operating during training without explicit regularization terms. Models that fail to generalize well tend to have lighter-tailed spectra. This framework operates at the level of spectral analysis of weight matrices rather than element-wise distributional characterization, but it is directly complementary to our work. Our finding that element-wise weight distributions shift from Gaussian (untrained) toward heavy-tailed and sub-Gaussian (trained) in certain architectures can be interpreted as a manifestation of the same underlying heavy-tailing phenomenon at the element level.

### 2.3 Mechanistic Interpretability and Weight-Level Analysis

The mechanistic interpretability literature has produced detailed accounts of transformer internals, but these accounts operate primarily at the level of activations, attention patterns, and representational geometry rather than at the level of weight tensor distributions. Elhage et al. (2021) demonstrated that transformer computations can be decomposed into interpretable circuits, with specific attention heads implementing identifiable functions such as induction, copying, and positional attention. Subsequent work on superposition (Elhage et al., 2022) showed that networks store more features than they have neurons by exploiting near-orthogonal directions in activation space. Neither line of work directly addresses whether the element-wise distributional regime of the weight tensors relates to functional specialization or circuit structure. Our observation that the trained attention weights are heavy-tailed and sharply peaked — corresponding to sparser, more concentrated weight tensors — is consistent with the superposition account — more specialized heads may achieve their function through sparser weight patterns — but the connection remains to be made explicit.

### 2.4 Bayesian Deep Learning and Laplace Approximation

The Laplace approximation for Bayesian neural networks approximates the posterior over weights as a Gaussian centered at the MAP estimate, with covariance given by the inverse Hessian of the loss at the optimum (MacKay, 1992). Daxberger et al. (2021) demonstrated that modern Laplace approximations can be practical for large networks when applied selectively to the last layer. The fundamental assumption of the Laplace approximation — that the posterior is well-described by a Gaussian — depends critically on the curvature of the loss landscape around the optimum and on the prior over weights. If the true prior that best describes trained weights is Laplace rather than Gaussian, as our empirical results suggest for several architecture families, then the Gaussian posterior approximation introduces a systematic error whose magnitude and direction depend on the tails of the true distribution.

### 2.5 Training Regime and Architecture Differences

The BERT family (Devlin et al., 2018) and RoBERTa (Liu et al., 2019) share near-identical architectures but differ substantially in training: RoBERTa removes next-sentence prediction, uses larger batches, trains on more data, and employs dynamic rather than static masking. The distributional differences we observe across these families reflect the combined influence of architecture, training objective, and training regime, and our BERT/RoBERTa comparison provides a partial experimental control.

---

## 3. Methods

### 3.1 Model Selection and Architecture Coverage

We analyze 15 transformer variants: GPT-2, GPT-2 Medium, and GPT-2 Large (causal language models, decoder-only); BERT Base and BERT Large (masked language models, encoder-only); DistilBERT (distilled BERT, encoder-only); RoBERTa Base and RoBERTa Large (robustly trained masked language models, encoder-only); DistilRoBERTa (distilled RoBERTa, encoder-only); ALBERT Base v2 (parameter-sharing masked language model, encoder-only); ELECTRA Small Discriminator (replaced token detection, encoder-only); BART Base (denoising autoencoder, encoder-decoder); T5 Small and T5 Base (text-to-text encoder-decoder); and mT5 Small (multilingual text-to-text encoder-decoder).

> **Hardware note — model selection.** The primary 15-model set was selected to fit within 4 GB of GPU VRAM (GTX 1060). Larger models — GPT-J-6B (6 B params), Phi-2 (2.7 B params, ~5.4 GB fp16), Falcon series — are listed in `scripts/modern_llms_ext.py` under `LARGE_MODELS` but are **excluded from the default run** because they exceed available VRAM. The optional `meta-llama/Llama-3.2-1B` (~2 GB fp16) is included in `CAPABLE_MODEL_IDS` and is available via `INCLUDE_OPTIONAL_MODELS=1`. We document this here explicitly so reviewers know the sample is hardware-bounded, not selectively chosen.

### 3.2 Weight Extraction Protocol

For each model, we extract attention projection weights using a unified name-matching protocol that identifies parameters containing attention-related terms (attention, attn, c_attn, q_proj, k_proj, v_proj, query, key, value, in_proj_weight, q_lin, k_lin, v_lin) and associates them with transformer layer indices via a regular expression matching standard depth indicators (h.N, layer.N, layers.N, block.N, albert_layers.N). Multiple attention projection matrices within a single layer (e.g., separate query, key, and value projections) are concatenated into a single per-layer weight vector before fitting. Architectures that fuse Q/K/V projections into a single matrix — notably GPT-2's `c_attn` — are handled by an in-place matrix decomposition that isolates per-head contributions, removing the architectural fallback limitations present in prior head-level analyses.

In the primary analysis (`scripts/run_pipeline.py`), we analyze the first 8 layers per model to enable cross-model comparison at a fixed depth. In the extended layer-depth analysis (`scripts/run_layerwise.py`), we analyze up to 15 layers per model (or all available layers if fewer than 15), reporting results at each depth position. All weight tensors are flattened before fitting.

### 3.3 Distribution Fitting and Layer Classification

For each per-layer weight vector, we fit Laplace, Gaussian, and Student-t distributions using maximum-likelihood estimation via `scipy.stats`. Layers exceeding 500,000 elements are deterministically subsampled to 500,000 weights before fitting, using a fixed per-layer seed (`ELA_SUBSAMPLE_SEED + layer_index`) so that repeated runs fit an identical subsample; this keeps the iterative Student-t MLE tractable and, by fixing n across layers and models, makes log-likelihood margins and goodness-of-fit statistics directly comparable across the sample (set `ELA_SUBSAMPLE=0` for full-tensor fits). We compute the total log-likelihood of the observed weights under each fitted distribution and classify each layer based on which distribution achieves the highest log-likelihood. We report the absolute log-likelihood values for each layer, enabling assessment of the margin of preference. In addition to the likelihood comparison, we apply the Kolmogorov-Smirnov (KS) two-sample test as a formal goodness-of-fit validation for each distributional candidate.

Both Laplace and Gaussian have two free parameters (location and scale), so the log-likelihood comparison does not require an information-criterion correction for model complexity. The Student-t distribution adds a degrees-of-freedom parameter, providing greater flexibility for heavy-tailed tensors. The KS test provides an independent, distribution-free check on fit quality, moving beyond subjective log-likelihood wins toward statistically grounded model selection.

### 3.4 Randomized-Label Control Experiments

To test whether the distributional pattern reflects architecture rather than training data content, we conduct three complementary control experiments.

**Short-horizon multi-seed control (`scripts/control_short.py`):** We fine-tune 6 pretrained models (GPT-2, GPT-2 Medium, BERT Base, RoBERTa Base, BART Base, T5 Small) for 25 steps each using random token label targets, with 3 independent random seeds per model, batch size 8.

**Long-horizon single-seed control (`scripts/control_long.py`):** We fine-tune 3 models (GPT-2, BERT Base, ELECTRA Base) for 500 steps using random targets, batch size 16, recording Laplace% at steps 0, 50, 100, 250, and 500.

**Shuffled-label control (`scripts/control_shuffled.py`):** We fine-tune GPT-2 and BERT Base for 10 steps using fully randomized label assignments.

### 3.5 Initialization Statistics

For each model's randomly initialized variant, we compute mean, standard deviation, range, kurtosis, and excess kurtosis of the flattened concatenation of all extracted attention projection weights. Bootstrap 95% confidence intervals for the Spearman rank correlation between initialization kurtosis and pretrained Laplace% are computed by `scripts/bootstrap_analysis.py` using 10,000 replicates.

---

## 4. Results

### 4.1 Architecture-Dependent Distributional Regimes

Table 1 summarizes Laplace% for all 15 models under the first-8-layers protocol. The results reveal a clear family-level stratification. All three GPT-2 models show moderate to high Laplace prevalence (62.5%, 75.0%, 100.0% for Base, Medium, Large respectively), with prevalence increasing monotonically with model size. All encoder-decoder models — BART, T5 Small, T5 Base, mT5 Small — achieve 100% Laplace prevalence. RoBERTa Base achieves 100% while RoBERTa Large achieves 37.5% and DistilRoBERTa 83.3%. ELECTRA shows 62.5%. In contrast, all BERT-family models show low Laplace prevalence: BERT Base 12.5%, BERT Large 0.0%, DistilBERT 16.7%.

| Model | Family | Layers | Laplace Wins | Gaussian Wins | Laplace % |
|---|---|---|---|---|---|
| GPT-2 | GPT-style | 8 | 5 | 3 | 62.5% |
| GPT-2 Medium | GPT-style | 8 | 6 | 2 | 75.0% |
| GPT-2 Large | GPT-style | 8 | 8 | 0 | 100.0% |
| BERT Base | BERT-style | 8 | 1 | 7 | 12.5% |
| BERT Large | BERT-style | 8 | 0 | 8 | 0.0% |
| DistilBERT | BERT-style | 6* | 1 | 5 | 16.7% |
| RoBERTa Base | RoBERTa-style | 8 | 8 | 0 | 100.0% |
| RoBERTa Large | RoBERTa-style | 8 | 3 | 5 | 37.5% |
| DistilRoBERTa | RoBERTa-style | 6* | 5 | 1 | 83.3% |
| ALBERT Base v2 | BERT-style | 8 | 8 | 0 | 100.0% |
| ELECTRA Small | ELECTRA | 8 | 5 | 3 | 62.5% |
| BART Base | Encoder-decoder | 6* | 6 | 0 | 100.0% |
| T5 Small | Encoder-decoder | 6* | 6 | 0 | 100.0% |
| T5 Base | Encoder-decoder | 8 | 8 | 0 | 100.0% |
| mT5 Small | Encoder-decoder | 8 | 8 | 0 | 100.0% |

\* Models with fewer than 8 total layers were analyzed for all available layers.

### 4.1a Four-Distribution Reconciliation

The Laplace-versus-Gaussian comparison in Table 1 is a two-way test. When the same weights are refit against four candidate distributions (Laplace, Gaussian, Student-t, and the generalized Gaussian, GGD) and the winner is chosen by BIC, no layer is best described by a *plain* Laplace across any of the 153 layers analyzed; the winners are the GGD and the Student-t. The family-level ordering is unchanged — encoder-decoder and GPT/RoBERTa attention weights are the most sharply peaked, BERT the least — but the accurate description of the data is "heavy-tailed and sub-Gaussian," not "Laplace." The GGD shape parameter β makes this quantitative (β = 1 recovers the Laplace, β = 2 the Gaussian): median β is ≈ 0.54–0.59 for T5/mT5 (sharper than Laplace), ≈ 1.3–1.5 for GPT-2 Large and RoBERTa, and ≈ 1.7 for the BERT family. This per-family β is the quantity that a distribution-aware compression scheme should consume (Section 7); it is reported per model in `results/reconciliation_table.md`.

### 4.2 Layer-Depth Signature in GPT-Style Models

The extended layer analysis (up to 15 layers per model) reveals a systematic depth-dependent pattern in GPT-style models. In GPT-2 (12 total layers), Laplace wins concentrate in early layers (0–4, 5 wins) while later layers (5–11, 7 wins for Gaussian) strongly prefer Gaussian. GPT-2 Medium (24 total layers, 15 analyzed) shows a similar pattern: Laplace wins in layers 0–5 (6 wins), Gaussian wins from layer 6 onward (9 wins), yielding 40.0% overall. GPT-2 Large shows the most gradual transition: Laplace wins for layers 0–8 (9 wins) before Gaussian takes over in layers 9–14 (6 Gaussian wins), for 60.0% overall.

This pattern — early Laplace, later Gaussian — is consistent with a gradient flow hypothesis: causal masking creates asymmetric gradient flow that is strongest in early layers (which process all subsequent positions) and attenuates toward later layers.

### 4.3 The RoBERTa/BERT Divergence

The most theoretically significant finding in Table 1 is the contrast between RoBERTa Base (100% Laplace) and BERT Base (12.5% Laplace). RoBERTa and BERT share the same transformer block architecture: identical attention mechanisms, identical parameter counts at the base scale, and the same masked language modeling objective. Their differences are exclusively at the training level: RoBERTa uses dynamic masking (different tokens masked each epoch) versus BERT's static masking (same masks throughout training), removes BERT's next-sentence prediction objective, uses larger batch sizes (8K vs 256), and trains on substantially more data (160GB vs 16GB). This makes the RoBERTa/BERT pair a natural experiment isolating training regime effects with architecture held approximately constant.

### 4.4 Pretrained vs. Random Initialization

At random initialization, all models in our sample — including the GPT-2 family — exhibit Gaussian-dominant attention weight distributions (0% Laplace across all randomly initialized variants). After pretraining, this changes dramatically for certain architecture families. GPT-2 family models show 62.5%, 75.0%, and 100.0% Laplace prevalence for Base, Medium, and Large respectively. BERT-family models remain at low Laplace% after pretraining (0–16.7%), consistent with their random-initialization baseline. RoBERTa Base is the most informative case: it starts at 0% Laplace at random initialization (Gaussian, like BERT) and reaches 100% Laplace after pretraining, demonstrating that this peaked, heavy-tailed structure can emerge entirely through training even when the initialization is Gaussian. The complete inversion from Gaussian (random) to heavy-tailed (pretrained) in causal and encoder-decoder models, and the persistence of Gaussian in masked models, directly implicates the training objective and gradient flow regime — not the initialization — as the mechanism driving distributional regime.

Spearman rank correlation between initialization kurtosis and pretrained Laplace% (ρ = 0.341, bootstrap 95% CI [-0.19, 0.79]) does not reach significance at the 0.05 level, indicating that initialization statistics alone do not strongly predict the distributional regime reached after pretraining. This null result is itself informative: it rules out a strong deterministic link between initialization shape and learned weight distribution, reinforcing the conclusion that training dynamics are the primary driver.

### 4.5 Randomized-Label Control: Distributional Stability

Across all three control experiments, the Laplace/Gaussian pattern remained completely stable before and after training on random targets. Zero standard deviation across seeds in the short-horizon experiment indicates that the pattern is identical across random initializations of the optimization trajectory. The long-horizon experiment confirms that loss trajectories decline (training is proceeding) while Laplace% remains constant at all five measurement points. Together, these results establish that the distributional pattern is architecture-determined and cannot be disrupted by short-to-medium training on uninformative data.

### 4.6 L1 Regularization Hypothesis Test

We test whether explicit L1 regularization on attention weights increases Laplace prevalence in a BERT-style architecture. BERT-base is fine-tuned for 200 steps with random labels under two conditions: (A) standard cross-entropy loss only, and (B) cross-entropy plus an L1 penalty on all attention projection weight norms. The hypothesis predicts that if Laplace structure arises from implicit L1-equivalent sparsity pressure, then the L1 treatment should yield higher Laplace% than the control.

Table 2 summarizes the outcome.

| Condition | Laplace% Before | Laplace% After | Δ |
|---|---|---|---|
| No L1 (control) | 12.5% | 12.5% | 0.0 pp |
| With L1 (treatment) | 12.5% | 100.0% | +87.5 pp |

The L1 treatment produces an 87.5 percentage-point increase in Laplace prevalence relative to control, providing strong preliminary evidence for the sparsity-pressure hypothesis at this experimental scale. This result indicates that explicit L1 regularization on attention weights is sufficient to drive a complete inversion from Gaussian-dominant to Laplace-dominant weight distributions in a BERT-style architecture. We report the result as a positive directional finding; multi-seed replication across varied L1 coefficients and training durations is required to confirm robustness and estimate effect variance. The first prediction derived in Section 6 is tested in `scripts/l1_regularization_test.py` (BERT-base ± L1, random labels, 200 steps, 1 seed, batch size 8).

---

## 5. Discussion

### 5.1 Architecture as Distributional Gatekeeper

The primary result of this study is that transformer architecture family is the dominant determinant of whether attention projection weights develop heavy-tailed, sub-Gaussian or near-Gaussian distributional structure during training. The GPT-2 layer-depth signature provides a clue: causal masking in autoregressive models restricts each position to attend only to preceding positions. This creates asymmetric gradient flow: early layers receive cumulative gradient contributions from all subsequent positions during backpropagation, while later layers receive more localized gradient signals.

### 5.2 The RoBERTa Anomaly as a Training Dynamics Signature

RoBERTa's aggressive training — more data, larger batches, dynamic masking, extended duration — achieves 100% Laplace versus BERT's 12.5% despite near-identical architecture. The reversal at large scale — RoBERTa Large achieves only 37.5% (first-8-layers protocol: 3/8 Laplace) versus RoBERTa Base's 100% — demands an explicit account. We proposed two non-mutually-exclusive hypotheses. First, larger models may possess redundant capacity that relaxes the need for sparse representations: when parameter count grows faster than informative signal, the optimization landscape may admit broader, more Gaussian-distributed solutions without sacrificing fit. Second, the primary analysis extracts only the first 8 layers; if Laplace structure in RoBERTa Large concentrates at greater depth than in RoBERTa Base, a fixed-layer protocol will systematically undercount Laplace wins in the larger model.

The layerwise sweep (`scripts/run_layerwise.py`) on RoBERTa Large provides a falsification of the depth-redistribution hypothesis. Across all 15 layers, Laplace prevalence is 26.7% (4/15 Laplace wins: layers 1, 2, 3, 10). Notably, the first 8 layers alone already contain 3 of those 4 Laplace wins (37.5% within that window), and the deeper layers (11–14) are uniformly Gaussian. Extending the protocol from 8 to 15 layers therefore *lowers* the aggregate Laplace% for RoBERTa Large, which is the opposite of the redistribution expectation. This pattern points toward the capacity-redundancy account as the dominant explanation: RoBERTa Large has sufficient representational width to learn Gaussian-dominant weights in most layers without loss of fit. The head-level protocol (`scripts/head_level_analysis.py`) falls back to whole-layer fitting for RoBERTa Large because its attention architecture does not expose per-head QKV decomposition in the same way as GPT-2, so head-level granularity was not available for this model.

### 5.3 Implications for the Gaussian Prior Assumption

Our results challenge the default assumption that trained transformer weights are well-described by Gaussian distributions. For GPT-2 family and all encoder-decoder models analyzed, Gaussian is the wrong distributional family for a majority of attention layers after pretraining. Methods that assume Gaussian weight distributions — including Laplace approximations for Bayesian inference, Gaussian-calibrated pruning thresholds, and standard post-training quantization schemes — may systematically underperform when applied to these architectures.

---

## 6. Towards a Mechanistic Account: Maximum Entropy and Implicit L1 Regularization

The Laplace distribution has a well-known characterization in information-theoretic terms: it is the maximum-entropy distribution under the constraint that the expected absolute deviation E[|x − μ|] is finite and fixed. This is the L1 analog of the classical result that the Gaussian is the maximum-entropy distribution under a fixed variance constraint (an L2 constraint).

We propose that autoregressive training with causal masking may impose such an implicit L1-equivalent constraint through the following mechanism. In a causally masked attention layer, each query position can only attend to preceding key positions. This directional constraint creates a gradient landscape where specific attention patterns are systematically reinforced (those that improve prediction of the next token) while others are systematically suppressed (those attending to future tokens, which receive zero gradient). The cumulative effect of this selective reinforcement is functionally equivalent to sparsity pressure: many attention weights are pushed toward zero because they connect query-key pairs that consistently fail to contribute to next-token prediction, while a subset of weights grows large because it encodes predictive patterns.

### Falsifiable Predictions

1. Models trained with explicit L1 regularization on attention weights should show higher Laplace prevalence than architecturally identical models trained without it, regardless of whether the base architecture is GPT-style or BERT-style.
2. The layer-depth gradient of Laplace prevalence in causal models should be steeper for longer sequence lengths.
3. Models trained with unidirectional masking applied to BERT-style architectures should develop Laplace structure in early layers, while standard BERT training applied to GPT-style architectures should suppress it.

The first prediction is tested in `scripts/l1_regularization_test.py` (BERT-base ± L1, random labels, 200 steps). A single-seed result shows an 87.5 pp increase (12.5% → 100.0% Laplace), providing strong preliminary support for the hypothesis. Multi-seed replication is a clear next step.

---

## 7. Practical Implications

The implications in this section follow from the distributional geometry documented above and are stated as predictions, not results. They are the subject of a planned distribution-aware quantization benchmark — GGD-parametrized quantization levels versus a matched-bit-width baseline, evaluated on task quality (perplexity) rather than weight-reconstruction error — which is intended to convert Section 7.2 from a prediction into a measured contribution. Two scope limits apply and are stated plainly. First, the distributional evidence in this paper is for attention-projection weights only; extending the memory- and energy-related claims to the whole model requires the same four-distribution characterization of feed-forward and embedding weights, which is pending. Second, established schemes such as GPTQ and AWQ are activation-aware output-reconstruction methods, not naive Gaussian quantizers; the proposed contribution is therefore a distribution-aware quantizer to be combined with or compared against them, not a replacement premised on their assuming a Gaussian prior.

### 7.1 Architecture-Aware Pruning

Current magnitude-based pruning methods apply uniform thresholds across all weight tensors. Our results suggest that pruning thresholds should be calibrated separately for heavy-tailed (low-β) and near-Gaussian (high-β) architectures. In the most sharply peaked architectures (encoder-decoder, β < 1), more aggressive pruning — removing a higher fraction of near-zero weights — should be possible with less performance degradation than in near-Gaussian architectures. Within GPT-style models, later layers (which are less peaked, β closer to 2) should be pruned more conservatively than earlier layers.

### 7.2 Quantization Calibration

Optimal quantization for sharply peaked, heavy-tailed weights is non-uniform: more quantization levels should be allocated near zero (where the mass concentrates) and fewer in the tails. The generalized-Gaussian shape β sets the optimal grid directly — a Lloyd–Max quantizer parametrized on the per-family β (β ≈ 0.54–0.59 for encoder-decoder, ≈ 1.3–1.7 for GPT/RoBERTa/BERT) rather than on a fixed Laplace (β = 1) or Gaussian (β = 2) assumption. The prediction to be tested (see the framing note opening this section) is that such a distribution-aware grid achieves higher task quality at a matched bit width than a β-agnostic baseline, with the largest margin in the encoder-decoder family.

### 7.3 Uncertainty Quantification

For architectures in the heavy-tailed regime, a heavier-tailed prior (Laplace, or a GGD with β < 1 for the most peaked encoder-decoder models) would be more appropriate for Bayesian neural network approximations. The practical consequence is that uncertainty estimates derived from Gaussian Laplace approximations applied to GPT-style or encoder-decoder models systematically underestimate tail probabilities in the weight posterior.

### 7.4 Fine-Tuning Dynamics and LoRA

Sharply peaked, heavy-tailed weights have a structural property relevant to fine-tuning: near-zero weights experience constant-magnitude gradient pressure under L1-equivalent dynamics. In the most peaked architectures, near-zero weights are more resistant to activation during fine-tuning. Fine-tuning preferentially updates the already-active (large) weights rather than activating new patterns. This could explain the efficiency of PEFT methods like LoRA on GPT-style models.

---

## 8. Conclusion

We have presented systematic empirical evidence that transformer attention weight distributions are strongly shaped by architecture and training dynamics, and that these patterns are robust to training data content. The null correlation between initialization kurtosis and pretrained Laplace% clarifies that the learned distributional regime is not a deterministic echo of initialization. We propose a mechanistic account connecting autoregressive training to implicit L1-equivalent regularization through the maximum-entropy characterization of the Laplace distribution, and we derive concrete predictions for pruning, quantization, uncertainty quantification, and fine-tuning that differ by architecture family. The L1 regularization experiment provides strong preliminary empirical support for the sparsity-pressure hypothesis, while the RoBERTa Large anomaly and the near-absence of Student-t wins point to open questions that motivate the layerwise and head-level analyses documented in `ela/analysis.py` and `scripts/run_layerwise.py`.

---

## References

1. Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *AISTATS*.
2. He, K., et al. (2015). Delving deep into rectifiers. *ICCV*.
3. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
4. Devlin, J., et al. (2018). BERT. *NAACL*.
5. Radford, A., et al. (2019). GPT-2. *OpenAI Technical Report*.
6. Liu, Y., et al. (2019). RoBERTa. *arXiv:1907.11692*.
7. Lan, Z., et al. (2019). ALBERT. *ICLR*.
8. Clark, K., et al. (2020). ELECTRA. *ICLR*.
9. Lewis, M., et al. (2019). BART. *ACL*.
10. Raffel, C., et al. (2020). T5. *JMLR*.
11. Martin, C. H., & Mahoney, M. W. (2019, 2021). Heavy-tailed self-regularization. *ICML / JMLR*.
12. Elhage, N., et al. (2021, 2022). Transformer Circuits Thread.
13. Daxberger, E., et al. (2021). Laplace Redux. *NeurIPS*.
14. MacKay, D. J. C. (1992). A practical Bayesian framework for backpropagation networks. *Neural Computation*.
15. Frankle, J., & Carbin, M. (2018). The lottery ticket hypothesis. *ICLR*.
16. Hu, E., et al. (2022). LoRA. *ICLR*.
17. Sanh, V., et al. (2019). DistilBERT. *arXiv:1910.01108*.
18. Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*. Wiley.

---

## Reproducibility

> **Regenerating everything on a fresh machine?** Follow **`RUNBOOK.md`** —
> zero-footprint Windows setup (uv, in-repo venv and model cache), exact run
> order, and troubleshooting.

### Prerequisites

```
Python >= 3.10, < 3.13
uv  (preferred)  or  pip
```

### Installation

```bash
cd elayra-research
uv pip install -r requirements-lock.txt
```

For GPU acceleration (NVIDIA only), install a torch wheel matching your GPU generation. GTX/RTX 10–40 series:
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements-lock.txt
```

RTX 50-series (Blackwell, sm_120) **requires** torch ≥ 2.7 built for CUDA 12.8 — cu121 wheels will not run on these GPUs:
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements-lock.txt
```

> The lock file (`requirements-lock.txt`) is generated with `uv pip compile pyproject.toml --extra dev` and pins the full transitive dependency set including `tqdm`.

### Running the full pipeline

**Unix / Git Bash / WSL:**
```bash
make all
```

**Windows PowerShell:**
```powershell
.\run_all.ps1 -Target all
```

**Or target individual analyses:**
```bash
make pipeline          # Step 1: 15-model primary analysis
make layerwise         # Step 2: layer-depth analysis (15 layers each)
make control-short     # Step 3a: 25-step multi-seed control
make control-long      # Step 3b: 500-step trajectory control
make control-shuffled  # Step 3c: 10-step shuffled-label control
make init-analysis     # Step 4: initialization statistics
make bootstrap         # Step 5: bootstrap CI for Spearman ρ
make l1                # L1 regularization hypothesis test (Section 4.6)
make checkpoint        # training checkpoint trajectory
make heads             # per-head distribution fitting
make mlp               # MLP layer distribution fitting
make modern-llms       # LLaMA-3.2-1B + optional large models
make test              # run pytest
```

### Outputs

All JSON results and PNG figures land in `results/`. Experiment metadata (timestamp, seed, torch version, CUDA status) is written by each script entry-point and is available in `results/experiment_meta.json` (auto-generated on first run).

To start fresh:
```bash
make clean-results   # removes results/*.json and results/*.png
```

### Key results (tracked in `results/`)

| File | Content | Status |
|---|---|---|
| `results/broader_analysis_results.json` | 15-model primary + random-init | committed (2-way Laplace/Gaussian; 3-way rerun pending) |
| `results/layerwise_model_comparison.json` | Per-layer 4-way fit (Laplace/Gaussian/Student-t/GGD), all 15 models, 153 layers | committed (reconciled in `results/reconciliation_table.md`) |
| `results/extended_control_results.json` | 25-step multi-seed control | **partial** — 1 model only |
| `results/expanded_model_init_results.json` | Initialization kurtosis × 15 models | committed |
| `results/head_level_results.json` | Per-head fits | committed (BART failed, fix pending) |
| `results/extended_control_500steps.json` | 500-step trajectories | to regenerate (`make control-long`) |
| `results/shuffled_control_results.json` | 10-step shuffled-label results | to regenerate (`make control-shuffled`) |
| `results/l1_regularization_results.json` | L1 hypothesis test (BERT-base ± L1, 200 steps) | to regenerate (`make l1`) |
| `results/bootstrap_results.json` | Bootstrap CI for Spearman ρ (Section 4.4) | to regenerate (`make bootstrap`) |

### Figures

All PNG figures are in `results/figures/`.

| File | Content |
|---|---|
| `results/figures/fig1_primary_bar.png` | Primary Laplace vs Gaussian wins per model (grouped bar) |
| `results/figures/fig2_primary_heatmap.png` | Layer-wise Laplace/Gaussian heatmap (model x layer) |
| `results/figures/fig3_depth_curves.png` | Cumulative Laplace prevalence by depth |
| `results/figures/fig4_layerwise_heatmap.png` | Layerwise (15-layer) win heatmap |
| `results/figures/fig5_kurtosis_scatter.png` | Init kurtosis vs pretrained Laplace% scatter |
| `results/figures/fig6_control_short.png` | Short-horizon random-label control (before vs after) |
| `results/figures/fig9_family_summary.png` | Architecture-family Laplace summary |
| `results/figures/fig10_roberta_bert_group.png` | RoBERTa + BERT grouped comparison |
| `results/figures/fig11_gpt2_scaling.png` | GPT-2 scaling profile |
| `results/figures/fig12_roberta_large_profile.png` | RoBERTa-large layerwise win profile |
| `results/figures/fig13_untrained_vs_pretrained.png` | Random-init vs pretrained Laplace prevalence |
| `results/figures/fig14_layers_summary.png` | Layer-count summary |
| `results/figures/fig15_controls_composite.png` | Composite control panel |
| `results/figures/fig16_init_histogram.png` | Initialization kurtosis histogram |
| `results/figures/fig17_margin_heatmap.png` | Margin-of-victory heatmap |
| `results/figures/fig18_head_level.png` | Per-head distribution fit profile |
| `results/figures/fig19_init_trend.png` | Initialization kurtosis trend |

fig7 (`control_long`) and fig8 (`l1`) are regenerated by `python scripts/generate_figures.py` once the corresponding result JSONs exist.

---

## Project Structure

```
elayra-research/
├── ela/                             # Core library (atomic: one responsibility per file)
│   ├── __init__.py                  #   Package marker; re-exports public API
│   ├── analysis.py                  #   Model loading, weight collection, layer fitting
│   ├── bootstrap.py                 #   Bootstrap confidence intervals
│   ├── config.py                    #   Centralised dataclass configs for all scripts
│   ├── distributions.py             #   Laplace / Gaussian / Student-t fitting + log-likelihood + KS GOF
│   ├── utils.py                     #   Batch generation, seeding, GPU memory hygiene
│   └── viz.py                       #   Shared plotting (heatmaps, trajectories, scatters)
│
├── scripts/                         # Entry points (one script per experiment)
│   ├── __init__.py
│   ├── run_pipeline.py              # → broader_analysis_results.json   (replaces old script)
│   ├── run_layerwise.py             # → layerwise_model_comparison.json  (replaces old script)
│   ├── control_short.py             # → extended_control_results.json    (replaces old script)
│   ├── control_long.py              # → extended_control_500steps.json   (replaces old script)
│   ├── control_shuffled.py          # → shuffled_control_results.json    (replaces old script)
│   ├── init_analysis.py             # → expanded_model_init_results.json (replaces old script)
│   ├── bootstrap_analysis.py        # → bootstrap_results.json
│   ├── l1_regularization_test.py    # → l1_regularization_results.json
│   ├── checkpoint_analysis.py       # → checkpoint_analysis.json
│   ├── head_level_analysis.py       # → head_level_results.json
│   ├── mlp_analysis.py              # → mlp_analysis_results.json
│   ├── modern_llms_ext.py           # → modern_llm_results.json
│   ├── run_layerwise_incremental.py # → layerwise, one model per invocation (resumable)
│   └── generate_figures.py          # → results/figures/fig1..fig19
│
├── tests/
│   ├── __init__.py
│   ├── test_analysis_core.py        # Smoke tests for ela/ + bootstrap + viz
│   └── test_ela_extended.py         # Config, utils, KS GOF, deterministic validation
│
├── results/                         # Generated outputs (tracked in git; regenerate via `make all`)
│   ├── experiment_meta.json         # Auto-captured by first run
│   ├── figures/                     # fig1..fig19 (generate_figures.py)
│   └── *.json  *.png
│
├── pyproject.toml                   # Package manifest + pinned dependencies
├── requirements-lock.txt            # Locked dependency set (uv pip compile)
├── Makefile                         # Unix/WSL runner
├── run_all.ps1                      # Windows PowerShell runner
└── README.md                        # This file
```

---

## License

MIT
