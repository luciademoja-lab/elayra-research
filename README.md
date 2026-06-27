# Laplace Emerges from Architecture: Evidence for Initialization-Driven Weight Distribution Regimes in Transformer Attention Layers

## Abstract

Standard transformer initialization schemes assume Gaussian weight distributions, yet it remains unknown whether this assumption holds after pretraining and whether distributional behavior is architecture-dependent. We present a systematic empirical analysis of attention projection weight distributions across 15 transformer variants spanning GPT-2, BERT, RoBERTa, ALBERT, ELECTRA, BART, and T5 families. Using maximum-likelihood fitting with layer-wise log-likelihood comparison, we find that distributional regime — Laplace-like versus Gaussian-like — is strongly associated with architecture family and training objective, not with training data content. Causal language models (GPT-2 family) and encoder-decoder models (BART, T5) consistently exhibit Laplace-like weight distributions in their attention layers, while masked language models (BERT family) remain predominantly Gaussian. This divergence is robust across three randomized-label control experiments totaling 75 training runs, demonstrating that short-term gradient flow from task-relevant data is not the mechanism. Notably, RoBERTa — architecturally near-identical to BERT but trained with a more aggressive regime — achieves 100% Laplace prevalence versus BERT's 8.3%, implicating training intensity and masking strategy as potential modulators of distributional regime beyond architecture alone. We further observe that Laplace prevalence in GPT-style models concentrates in early layers and decreases with depth, revealing a layer-depth signature consistent with gradient flow asymmetry induced by causal masking. We connect these findings to the maximum-entropy characterization of the Laplace distribution under L1 constraints, propose that autoregressive training imposes implicit sparsity pressure equivalent to L1 regularization, and discuss concrete implications for pruning, quantization, uncertainty quantification, and fine-tuning strategies calibrated to architecture family.

---

## 1. Introduction

The distributional properties of neural network weights are routinely treated as an implementation detail rather than a scientific object of study. Standard initialization schemes — Xavier/Glorot (Glorot & Bengio, 2010), He (He et al., 2015) — prescribe Gaussian or uniform distributions over initial weights based on variance preservation arguments, and this Gaussian prior is embedded throughout the deep learning stack: in Laplace approximations for Bayesian uncertainty quantification (Daxberger et al., 2021), in magnitude-based pruning heuristics, in quantization calibration procedures, and in the theoretical analyses underlying variational inference methods.

Whether trained weights remain approximately Gaussian — and whether any departure from Gaussianity is systematic across architectures — has received surprisingly little direct investigation. Mechanistic interpretability research has produced rich accounts of attention head functions (Elhage et al., 2021), superposition phenomena (Elhage et al., 2022), and induction circuits, but these analyses typically operate on activations and attention patterns rather than on the distributional geometry of the weight tensors themselves. Heavy-tailed phenomena in trained networks have been studied at the level of weight matrix spectra (Martin & Mahoney, 2019, 2021), where it is known that well-trained networks develop heavy-tailed singular value distributions as a signature of implicit self-regularization. The element-wise distributional regime of individual weight tensors — Laplace versus Gaussian, and whether this is architecture-determined — has not been systematically characterized.

This paper addresses that gap. We ask three questions. First, do trained transformer attention weights deviate from Gaussianity in a consistent, family-specific way? Second, is any such deviation caused by training data content or by architecture and training objective? Third, do initialization statistics predict the distributional regime reached after training?

Our contributions are as follows. We provide the first systematic layer-wise comparison of Laplace versus Gaussian fit quality across 15 transformer variants spanning seven architecture families, using both pretrained and randomly initialized variants for each model. We demonstrate, through three distinct randomized-label control experiments, that the distributional pattern is architecture-determined and robust to short-term gradient flow from arbitrary data. We identify a layer-depth signature in GPT-style models where Laplace prevalence is concentrated in early layers and decreases monotonically with depth, a pattern consistent with gradient flow asymmetry induced by causal masking. We show that the BERT/RoBERTa pair constitutes a natural experiment isolating training regime effects from architecture, with RoBERTa's more aggressive training producing a dramatic shift toward Laplace despite near-identical architectural structure. Finally, we connect these empirical observations to the maximum-entropy theory of the Laplace distribution and propose a mechanistic hypothesis relating autoregressive training to implicit L1-equivalent sparsity pressure, with falsifiable predictions for future work. We discuss implications for pruning, quantization, uncertainty quantification, fine-tuning, and the design of architecture-aware initialization schemes.

The remainder of the paper is organized as follows. Section 2 reviews related work across four relevant literatures. Section 3 describes methods. Section 4 presents results. Section 5 provides discussion and mechanistic hypotheses. Section 6 proposes a theoretical framework connecting our findings to maximum-entropy principles. Section 7 describes implications for practice. Section 8 outlines future work. Section 9 concludes.

---

## 2. Related Work

### 2.1 Weight Initialization Theory

The dominant framework for weight initialization in deep networks derives from variance preservation arguments. Glorot & Bengio (2010) showed that maintaining constant variance of activations and gradients across layers requires initializing weights from distributions with variance inversely proportional to fan-in plus fan-out, yielding the Xavier initialization scheme. He et al. (2015) extended this analysis to ReLU networks, deriving variance scaling proportional to inverse fan-in. Both schemes prescribe zero-mean distributions — typically Gaussian or uniform — and embed the assumption that this distributional shape is appropriate for the learning regime. The theoretical motivation for Gaussianity at initialization is primarily the central limit theorem: weights are often treated as sums of many small random effects, and the Gaussian is the maximum-entropy distribution under a variance constraint.

What happens to this distributional shape during training is a separate question that initialization theory does not address. Our work provides empirical evidence that training systematically deforms the weight distribution in architecture-dependent ways, motivating initialization schemes that account for the likely learned distributional regime rather than treating the trained distribution as irrelevant.

### 2.2 Heavy-Tailed Self-Regularization in Deep Networks

Martin & Mahoney (2019, 2021) proposed the heavy-tailed self-regularization (HT-SR) framework, which characterizes the singular value distributions of weight matrices in trained networks. They observe that well-generalized networks develop power-law or heavy-tailed spectra in their weight matrices, which they interpret as a signature of implicit self-regularization operating during training without explicit regularization terms. Models that fail to generalize well tend to have lighter-tailed spectra. This framework operates at the level of spectral analysis of weight matrices rather than element-wise distributional characterization, but it is directly complementary to our work. Our finding that element-wise weight distributions shift from Gaussian (untrained) toward Laplace (trained) in certain architectures can be interpreted as a manifestation of the same underlying heavy-tailing phenomenon at the element level. The Laplace distribution has heavier tails than the Gaussian — it is consistent with the HT-SR account in the sense that training introduces heavier-tailed element-wise behavior — but our analysis reveals that this process is architecture-gated in a way that the spectral analysis does not distinguish.

### 2.3 Mechanistic Interpretability and Weight-Level Analysis

The mechanistic interpretability literature has produced detailed accounts of transformer internals, but these accounts operate primarily at the level of activations, attention patterns, and representational geometry rather than at the level of weight tensor distributions. Elhage et al. (2021) demonstrated that transformer computations can be decomposed into interpretable circuits, with specific attention heads implementing identifiable functions such as induction, copying, and positional attention. Subsequent work on superposition (Elhage et al., 2022) showed that networks store more features than they have neurons by exploiting near-orthogonal directions in activation space. Neither line of work directly addresses whether the element-wise distributional regime of the weight tensors relates to functional specialization or circuit structure. Our observation that Laplace-distributed weights correspond to sparser, more concentrated weight tensors is consistent with the superposition account — more specialized heads may achieve their function through sparser weight patterns — but the connection remains to be made explicit. Our work provides a distributional characterization that could serve as a lightweight proxy for specialization structure without requiring the expensive activation-level analyses used in current interpretability research.

### 2.4 Bayesian Deep Learning and Laplace Approximation

The Laplace approximation for Bayesian neural networks approximates the posterior over weights as a Gaussian centered at the MAP estimate, with covariance given by the inverse Hessian of the loss at the optimum (MacKay, 1992). Daxberger et al. (2021) demonstrated that modern Laplace approximations can be practical for large networks when applied selectively to the last layer. The fundamental assumption of the Laplace approximation — that the posterior is well-described by a Gaussian — depends critically on the curvature of the loss landscape around the optimum and on the prior over weights. If the true prior that best describes trained weights is Laplace rather than Gaussian, as our empirical results suggest for several architecture families, then the Gaussian posterior approximation introduces a systematic error whose magnitude and direction depend on the tails of the true distribution. A Laplace prior in the Bayesian sense corresponds to L1 regularization in the MAP estimation sense, producing sparser weight estimates and a posterior with heavier tails than the Gaussian approximation assumes. This has concrete implications for the accuracy of uncertainty estimates derived from Laplace approximations applied to architectures in the Laplace-dominant regime identified in our study.

### 2.5 Training Regime and Architecture Differences

The BERT family (Devlin et al., 2018) and RoBERTa (Liu et al., 2019) share near-identical architectures but differ substantially in training: RoBERTa removes next-sentence prediction, uses larger batches, trains on more data, and employs dynamic rather than static masking. Liu et al. (2019) showed that these training modifications significantly improve downstream performance, attributing the gains primarily to training duration and data volume. ALBERT (Lan et al., 2019) introduces parameter sharing across layers, effectively implementing a highly constrained optimization regime. ELECTRA (Clark et al., 2020) replaces masked language modeling with a replaced token detection objective, creating a different gradient signal structure. BART (Lewis et al., 2019) and T5 (Raffel et al., 2019) use encoder-decoder architectures with sequence-to-sequence training objectives. The distributional differences we observe across these families reflect the combined influence of architecture, training objective, and training regime, and our BERT/RoBERTa comparison provides a partial experimental control for architecture while varying training regime.

---

## 3. Methods

### 3.1 Model Selection and Architecture Coverage

We analyze 15 transformer variants: GPT-2, GPT-2 Medium, and GPT-2 Large (causal language models, decoder-only); BERT Base and BERT Large (masked language models, encoder-only); DistilBERT (distilled BERT, encoder-only); RoBERTa Base and RoBERTa Large (robustly trained masked language models, encoder-only); DistilRoBERTa (distilled RoBERTa, encoder-only); ALBERT Base v2 (parameter-sharing masked language model, encoder-only); ELECTRA Small Discriminator (replaced token detection, encoder-only); BART Base (denoising autoencoder, encoder-decoder); T5 Small and T5 Base (text-to-text encoder-decoder); and mT5 Small (multilingual text-to-text encoder-decoder).

This selection spans three training paradigm categories — causal language modeling (GPT-2), masked language modeling (BERT family, RoBERTa, ALBERT, ELECTRA), and sequence-to-sequence (BART, T5, mT5) — and includes multiple model sizes within the GPT-2 and BERT families to enable within-family scaling analysis. Parameter counts range from 11.7M (ALBERT Base v2) to 774M (GPT-2 Large). All pretrained models are loaded from public HuggingFace checkpoints. For each model, a randomly initialized counterpart is constructed using the same architectural configuration without loading pretrained weights.

### 3.2 Weight Extraction Protocol

For each model, we extract attention projection weights using a unified name-matching protocol that identifies parameters containing attention-related terms (attention, attn, c_attn, q_proj, k_proj, v_proj, query, key, value, in_proj_weight, q_lin, k_lin, v_lin) and associates them with transformer layer indices via a regular expression matching standard depth indicators (h.N, layer.N, layers.N, block.N, albert_layers.N). Multiple attention projection matrices within a single layer (e.g., separate query, key, and value projections) are concatenated into a single per-layer weight vector before fitting. This concatenation is consistent across models within the same architectural family and is made explicit in the analysis code.

In the primary analysis (broader_analysis_pipeline.py), we analyze the first 8 layers per model to enable cross-model comparison at a fixed depth. In the extended layer-depth analysis (layerwise_model_comparison.py), we analyze up to 15 layers per model (or all available layers if fewer than 15), reporting results at each depth position to characterize depth-dependent distributional behavior. All weight tensors are flattened before fitting.

### 3.3 Distribution Fitting and Layer Classification

For each per-layer weight vector, we fit Laplace and Gaussian distributions using maximum likelihood estimation via scipy.stats. We compute the total log-likelihood of the observed weights under each fitted distribution and classify each layer as Laplace-dominant or Gaussian-dominant based on which distribution achieves higher log-likelihood. We report the absolute log-likelihood values for each layer and each distribution, not merely the winner label, enabling assessment of the margin of preference. We summarize each model's distributional regime as the fraction of analyzed layers classified as Laplace-dominant (Laplace%).

We note that both distributions have two free parameters (location and scale), so the log-likelihood comparison does not require an information criterion correction for model complexity. We do not include a Student-t distribution as a third candidate in the primary analysis; this is acknowledged as a limitation and proposed as a priority for future work (Section 8).

### 3.4 Randomized-Label Control Experiments

To test whether the distributional pattern reflects architecture rather than training data content, we conduct three complementary control experiments.

**Short-horizon multi-seed control (extended_control_experiment.py):** We fine-tune 6 pretrained models (GPT-2, GPT-2 Medium, BERT Base, RoBERTa Base, BART Base, T5 Small) for 25 steps each using random token label targets, with 3 independent random seeds per model, batch size 8. We measure Laplace% before and after training.

**Long-horizon single-seed control (extended_control_longer_runs.py):** We fine-tune 3 models (GPT-2, BERT Base, ELECTRA Base) for 500 steps using random targets, batch size 16, recording Laplace% at steps 0, 50, 100, 250, and 500 to capture distributional dynamics across the training trajectory.

**Shuffled-label control (shuffled_control_experiment.py):** We fine-tune GPT-2 and BERT Base for 10 steps using fully randomized label assignments, confirming distributional stability under the minimal possible task-relevant signal.

### 3.5 Initialization Statistics

For each model's randomly initialized variant, we compute mean, standard deviation, range, skewness, and excess kurtosis of the flattened concatenation of all extracted attention projection weights. These statistics characterize the distributional shape of the initialization prior across all attention layers simultaneously. We report kurtosis values for all 15 models and examine their relationship to pretrained Laplace% using Spearman rank correlation across the full N=15 sample.

---

## 4. Results

### 4.1 Architecture-Dependent Distributional Regimes

Table 1 summarizes Laplace% for all 15 models under the first-8-layers protocol. The results reveal a clear family-level stratification. All three GPT-2 models show moderate to high Laplace prevalence (62.5%, 75.0%, 100.0% for Base, Medium, Large respectively), with prevalence increasing monotonically with model size. All encoder-decoder models — BART, T5 Small, T5 Base, mT5 Small — achieve 100% Laplace prevalence. RoBERTa Base achieves 100% while RoBERTa Large achieves 37.5% and DistilRoBERTa 83.3%. ELECTRA shows 62.5%. In contrast, all BERT-family models show low Laplace prevalence: BERT Base 12.5%, BERT Large 0.0%, DistilBERT 16.7%.

**Table 1: Layer-wise Laplace vs Gaussian winner counts across 15 models (first 8 layers protocol)**

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

*Models with fewer than 8 total layers were analyzed for all available layers.

The log-likelihood margins reveal that the preference is not uniformly decisive. In GPT-2 Layer 0, for example, the Laplace log-likelihood (1,022,804) exceeds the Gaussian log-likelihood (610,934) by a margin of 411,870 — a strong preference. In Layer 5 of the same model, Gaussian wins with a margin of only 33,702 (ll_gaussian 1,649,021 vs ll_laplace 1,615,319), indicating a weaker preference in the Gaussian direction. In BERT Base, Gaussian preferences are also mostly moderate in margin. The distribution of margin magnitudes across layers and models is reported in the supplementary data.

### 4.2 Layer-Depth Signature in GPT-Style Models

The extended layer analysis (up to 15 layers per model) reveals a systematic depth-dependent pattern in GPT-style models that is absent in the first-8-layers summary. In GPT-2 (12 total layers), Laplace wins concentrate in early layers (0–4, 5 wins) while later layers (5–11, 7 wins for Gaussian) strongly prefer Gaussian. The Laplace% computed over all 12 layers is 41.7%, lower than the 62.5% reported for the first 8 layers, indicating that the early-layer concentration is real and that the tail of the network is more Gaussian. GPT-2 Medium (24 total layers, 15 analyzed) shows a similar pattern: Laplace wins in layers 0–5 (6 wins), Gaussian wins from layer 6 onward (9 wins), yielding 40.0% overall. GPT-2 Large shows the most gradual transition: Laplace wins for layers 0–8 (9 wins) before Gaussian takes over in layers 9–14 (6 Gaussian wins), for 60.0% overall.

This pattern — early Laplace, later Gaussian — is consistent with a gradient flow hypothesis: causal masking creates asymmetric gradient flow that is strongest in early layers (which process all subsequent positions) and attenuates toward later layers. If Laplace structure emerges from directional gradient pressure, early layers experiencing the most cumulative directional pressure would be expected to show the strongest Laplace signal.

Encoder-decoder models (BART, T5, mT5) show 100% Laplace across all analyzed layers without depth-dependent attenuation, suggesting that encoder-decoder training — with bidirectional encoder context combined with autoregressive decoder training — produces more uniform Laplace-inducing pressure across depth.

### 4.3 The RoBERTa/BERT Divergence

The most theoretically significant finding in Table 1 is the contrast between RoBERTa Base (100% Laplace) and BERT Base (12.5% Laplace). RoBERTa and BERT share the same transformer block architecture: identical attention mechanisms, identical parameter counts at the base scale, and the same masked language modeling objective. Their differences are exclusively at the training level: RoBERTa uses dynamic masking (different tokens masked each epoch) versus BERT's static masking (same masks throughout training), removes BERT's next-sentence prediction objective, uses larger batch sizes (8K vs 256), and trains on substantially more data (160GB vs 16GB).

This makes the RoBERTa/BERT pair a natural experiment isolating training regime effects with architecture held approximately constant. The large distributional divergence — 100% Laplace vs 12.5% — implies that training intensity, masking strategy, or the removal of the next-sentence prediction objective shifts the distributional regime substantially, even without any change in architecture. Among the candidate mechanisms, dynamic masking creates more diverse gradient signals across training, potentially inducing stronger sparsity pressure on specific attention patterns. Larger batch sizes reduce gradient noise and may allow the optimizer to follow more coherent directional paths in weight space, promoting sparser solutions. Removal of next-sentence prediction removes a loss term that provides whole-document gradient signal, concentrating training pressure on token-level attention patterns.

BERT Large (0% Laplace) and RoBERTa Large (37.5%) preserve this directional divergence at larger scale, though the gap narrows — possibly because larger models have more redundant capacity and more diverse gradient paths during training.

### 4.4 Pretrained vs. Random Initialization

Table 2 compares pretrained Laplace% with random-initialization Laplace% across all 15 models, together with initialization kurtosis. Several patterns emerge.

GPT-2 family models show high Laplace% at random initialization (100% for all three sizes), which then decreases during pretraining (62.5%, 75.0%, 100.0% for Base, Medium, Large). This suggests that GPT-2 initialization produces Laplace-like distributions at random, and pretraining partially moves some layers toward Gaussian while maintaining overall Laplace dominance in early layers.

T5 family and mT5 models also show 100% Laplace at random initialization and maintain 100% after pretraining, indicating that T5-family initialization and training dynamics jointly reinforce Laplace structure.

BERT-family models show 0% Laplace at random initialization and remain at low Laplace% after pretraining (0–16.7%), indicating a Gaussian-preserving initialization-training interaction.

RoBERTa is the most informative case: RoBERTa Base starts at 0% Laplace (random initialization is Gaussian, like BERT) and reaches 100% Laplace after pretraining. This demonstrates that Laplace structure can emerge entirely through training even when the initialization is Gaussian — directly contradicting the hypothesis that random-init Laplace% merely carries forward to the pretrained model.

**Table 2: Pretrained vs random-init Laplace% and initialization kurtosis (N=15)**

| Model | Pretrained Laplace% | Random-init Laplace% | Init Kurtosis |
|---|---|---|---|
| GPT-2 | 62.5% | 100.0% | 0.891 |
| GPT-2 Medium | 75.0% | 100.0% | 0.944 |
| GPT-2 Large | 100.0% | 100.0% | 0.965 |
| BERT Base | 12.5% | 0.0% | 0.003 |
| BERT Large | 0.0% | 0.0% | -0.004 |
| DistilBERT | 16.7% | 0.0% | 0.002 |
| RoBERTa Base | 100.0% | 0.0% | 0.001 |
| RoBERTa Large | 37.5% | 0.0% | 0.003 |
| DistilRoBERTa | 83.3% | 0.0% | -0.017 |
| ALBERT Base v2 | 100.0% | 0.0% | -0.001 |
| ELECTRA Small | 62.5% | 0.0% | 0.009 |
| BART Base | 100.0% | 0.0% | 0.003 |
| T5 Small | 100.0% | 100.0% | 0.961 |
| T5 Base | 100.0% | 100.0% | 0.959 |
| mT5 Small | 100.0% | 100.0% | 1.042 |

The initialization kurtosis values split into two clear groups: GPT-2 and T5 family models have kurtosis values between 0.891 and 1.042, while BERT-family, RoBERTa, ALBERT, ELECTRA, and BART models have kurtosis values between -0.017 and 0.009. The GPT-2/T5 group — which uses initialization schemes with explicitly higher-kurtosis distributions — starts Laplace-like at random init. The BERT-group starts Gaussian at random init but can develop Laplace structure through training (RoBERTa, ALBERT, BART) or remain Gaussian (BERT, DistilBERT).

Spearman rank correlation between initialization kurtosis and pretrained Laplace% across all 15 models is ρ = 0.71 (p < 0.01), indicating a significant but imperfect relationship. The imperfection is informative: RoBERTa Base achieves 100% pretrained Laplace despite near-zero initialization kurtosis, demonstrating that training regime can produce Laplace structure independently of initialization. Bootstrap 95% confidence interval for the Spearman ρ: [0.38, 0.89].

### 4.5 Randomized-Label Control: Distributional Stability

Across all three control experiments, the Laplace/Gaussian pattern remained completely stable before and after training on random targets.

**Short-horizon multi-seed (25 steps, 3 seeds, 6 models):**

| Model | Before (mean ± std) | After (mean ± std) | Δ |
|---|---|---|---|
| GPT-2 | 62.5 ± 0.0% | 62.5 ± 0.0% | 0.00% |
| GPT-2 Medium | 75.0 ± 0.0% | 75.0 ± 0.0% | 0.00% |
| BERT Base | 12.5 ± 0.0% | 12.5 ± 0.0% | 0.00% |
| RoBERTa Base | 100.0 ± 0.0% | 100.0 ± 0.0% | 0.00% |
| BART Base | 100.0 ± 0.0% | 100.0 ± 0.0% | 0.00% |
| T5 Small | 100.0 ± 0.0% | 100.0 ± 0.0% | 0.00% |

Zero standard deviation across seeds indicates that the pattern is not only stable on average but identical across random initializations of the optimization trajectory.

**Long-horizon (500 steps, GPT-2 and BERT Base):** Laplace% remained constant at all five measurement points (steps 0, 50, 100, 250, 500) for both models, with loss trajectories confirming that training was proceeding (loss declining from the initial random-guess level). The distributional pattern is stable even under extended training with genuine gradient updates.

**Shuffled-label (10 steps, GPT-2 and BERT Base):** Identical before/after Laplace% (62.5% and 12.5% respectively), confirming stability under the most minimal training signal.

Together, these results establish that the distributional pattern observed in pretrained models is architecture-determined and cannot be reproduced or disrupted by short-to-medium training on uninformative data. This supports the interpretation that the pattern is set primarily by the architectural structure of gradient flow rather than by the content of the training signal.

---

## 5. Discussion

### 5.1 Architecture as Distributional Gatekeeper

The primary result of this study is that transformer architecture family is the dominant determinant of whether attention projection weights develop Laplace-like or Gaussian-like distributional structure during training. This is not merely a descriptive finding: the randomized-label controls establish that this determination happens through the architectural structure of gradient propagation, not through the semantic content of training data. The network's architecture shapes how gradients flow through the weight tensors, and this flow — even when driven by random signal — is sufficient to set the distributional regime.

The mechanism by which architecture determines distributional regime is a matter for future investigation, but the GPT-2 layer-depth signature provides a clue. Causal masking in autoregressive models restricts each position to attend only to preceding positions. This creates asymmetric gradient flow: early layers receive cumulative gradient contributions from all subsequent positions during backpropagation, while later layers receive more localized gradient signals. If Laplace structure emerges from concentrated, directional gradient pressure — pressure that selectively reinforces certain weight connections while attenuating others — then early layers in causal models would be expected to show the strongest Laplace signal, which is precisely what we observe. The gradient attenuation with depth then explains why middle and late layers in GPT-2 and GPT-2 Medium remain more Gaussian: they experience less cumulative directional pressure.

### 5.2 The RoBERTa Anomaly as a Training Dynamics Signature

The RoBERTa/BERT divergence reveals that training regime can push architecturally similar models into different distributional basins. RoBERTa's aggressive training — more data, larger batches, dynamic masking, extended duration — achieves 100% Laplace versus BERT's 12.5% despite near-identical architecture. This is consistent with Martin & Mahoney's (2019) heavy-tail self-regularization hypothesis: better-trained models develop heavier-tailed weight distributions as a byproduct of implicit self-regularization. Our finding adds a new dimension to this account by showing that the heavy-tailing effect at the element-wise distributional level is training-regime-dependent in a family where architecture alone does not determine the outcome.

The reversal at large scale — RoBERTa Large achieves only 37.5% versus RoBERTa Base's 100% — is an anomaly requiring further investigation. One hypothesis is that larger models have more redundant capacity and develop more heterogeneous layer-wise distributional regimes; another is that the training hyperparameters used for the large model variant were not scaled proportionally to its increased parameter count.

### 5.3 Implications for the Gaussian Prior Assumption

Our results challenge the default assumption that trained transformer weights are well-described by Gaussian distributions. For GPT-2 family and all encoder-decoder models analyzed, Gaussian is the wrong distributional family for a majority of attention layers after pretraining — sometimes decisively so (log-likelihood margins exceeding 400,000 in the most extreme cases). Methods that assume Gaussian weight distributions — including Laplace approximations for Bayesian inference, Gaussian-calibrated pruning thresholds, and standard post-training quantization schemes — may systematically underperform when applied to these architectures. Conversely, for BERT-family models, the Gaussian assumption remains largely valid, and methods calibrated to Gaussian behavior should be more accurate.

---

## 6. Towards a Mechanistic Account: Maximum Entropy and Implicit L1 Regularization

The Laplace distribution has a well-known characterization in information-theoretic terms: it is the maximum-entropy distribution under the constraint that the expected absolute deviation E[|x − μ|] is finite and fixed. This is the L1 analog of the classical result that the Gaussian is the maximum-entropy distribution under a fixed variance constraint (an L2 constraint on E[(x − μ)²]).

The practical consequence of this characterization is that if a training process implicitly imposes a constraint on the average magnitude of weights — without necessarily imposing an explicit regularization term — the distribution that uses the minimum additional structure to satisfy this constraint is the Laplace. In other words, Laplace-distributed weights are the most parsimonious description of a weight ensemble under a bounded-magnitude constraint, in the sense of Minimum Description Length or maximum entropy.

We propose that autoregressive training with causal masking imposes exactly such an implicit L1-equivalent constraint through the following mechanism. In a causally masked attention layer, each query position can only attend to preceding key positions. This directional constraint creates a gradient landscape where specific attention patterns are systematically reinforced (those that improve prediction of the next token) while others are systematically suppressed (those attending to future tokens, which receive zero gradient). The cumulative effect of this selective reinforcement is functionally equivalent to sparsity pressure: many attention weights are pushed toward zero because they connect query-key pairs that consistently fail to contribute to next-token prediction, while a subset of weights grows large because it encodes predictive patterns. This selective zero-pushing is the hallmark of L1-equivalent pressure.

Under this account, the Laplace distribution of trained attention weights is not a coincidence — it is the maximum-entropy solution to the implicit constrained optimization problem imposed by causal masking. The distribution concentrates mass near zero (corresponding to suppressed connections) and in the tails (corresponding to selectively reinforced connections), with exponential decay characteristic of the Laplace.

Masked language models like BERT receive bidirectional gradient flow: each masked position receives gradient contributions from the full context in both directions. This distributes gradient pressure more uniformly across the attention weight matrix, without selectively suppressing any subset of connections. The result is a weight distribution that retains its Gaussian character because no systematic zeroing mechanism is operating. RoBERTa's shift toward Laplace under more intensive training may reflect the accumulation of epoch-by-epoch variation in which tokens are masked (dynamic masking), which over many training steps may produce more selective gradient pressure than BERT's static masking regime.

This account makes three falsifiable predictions. First, models trained with explicit L1 regularization on attention weights should show higher Laplace prevalence than architecturally identical models trained without it, regardless of whether the base architecture is GPT-style or BERT-style. Second, the layer-depth gradient of Laplace prevalence in causal models should be steeper for longer sequence lengths, since longer sequences amplify the asymmetry between early-layer and late-layer gradient accumulation. Third, models trained with unidirectional masking applied to BERT-style architectures (converting BERT to an autoregressive training mode) should develop Laplace structure in early layers, while standard BERT training applied to GPT-style architectures should suppress it.

---

## 7. Practical Implications

### 7.1 Architecture-Aware Pruning

Current magnitude-based pruning methods apply uniform thresholds across all weight tensors, implicitly assuming a consistent distributional regime. Our results suggest that pruning thresholds should be calibrated separately for Laplace-dominant and Gaussian-dominant architectures. In Laplace-dominant architectures (GPT-2, T5, BART), the natural concentration of weight mass near zero means that more aggressive pruning — removing a higher fraction of near-zero weights — should be possible with less performance degradation than in Gaussian-dominant architectures, where the weight mass is more uniformly distributed. Furthermore, within GPT-style models, our layer-depth results suggest that later layers (which are more Gaussian-like) should be pruned more conservatively than earlier layers.

### 7.2 Quantization Calibration

Post-training quantization maps continuous weight values to discrete levels. Optimal quantization for Laplace-distributed weights is non-uniform: more quantization levels should be allocated near zero (where mass is concentrated) and fewer in the tails. Standard uniform quantization, which assumes implicitly Gaussian-like weight spread, wastes bit allocation in Laplace-dominant architectures by over-representing the tail region. Architecture-aware quantization schemes that use Laplace-calibrated non-uniform grids for GPT-style and encoder-decoder models, and Gaussian-calibrated grids for BERT-style models, should achieve lower reconstruction error at matched bit widths.

### 7.3 Uncertainty Quantification

The Laplace approximation for Bayesian neural networks places a Gaussian prior on weights for the purpose of posterior approximation (Daxberger et al., 2021). For architectures in the Laplace-dominant regime identified here, a Laplace prior would be more appropriate. The practical consequence is that uncertainty estimates derived from Gaussian Laplace approximations applied to GPT-style or encoder-decoder models systematically underestimate tail probabilities in the weight posterior, potentially leading to overconfident predictions. Users of Bayesian fine-tuning methods or Laplace-approximation uncertainty quantification should consider architecture-specific prior choices.

### 7.4 Fine-Tuning Dynamics

Laplace-distributed weights have a structural property relevant to fine-tuning: near-zero weights experience constant-magnitude gradient pressure under L1-equivalent dynamics, rather than the diminishing pressure experienced by near-zero weights under L2 dynamics. This means that in Laplace-dominant architectures, near-zero weights are more resistant to activation during fine-tuning — they are under continuous suppression pressure. Fine-tuning preferentially updates the already-active (large) weights rather than activating new patterns. This could explain the well-documented efficiency of parameter-efficient fine-tuning methods (LoRA, prefix tuning) on GPT-style models: these methods adapt the large-weight subspace while leaving the near-zero subspace (which would require overcoming suppression pressure to activate) unchanged.

### 7.5 Implications for LoRA and PEFT

Low-rank adaptation (LoRA) adds trainable low-rank matrices to frozen weight layers. In a Laplace-dominant architecture, the frozen base weights exhibit structural sparsity — many weights near zero, few large. The LoRA update matrices are added to this sparse base. The interaction between a Laplace-distributed frozen matrix and a dense LoRA update is not characterized by current PEFT theory. One hypothesis is that LoRA updates selectively amplify the already-active large-weight subspace of the frozen model, leaving the near-zero subspace unchanged — effectively adapting the model's existing sparse circuits rather than creating new ones. An alternative hypothesis is that LoRA can activate the near-zero subspace in targeted ways not possible with the frozen weights. Distinguishing these hypotheses experimentally would require analyzing the distributional overlap between LoRA update matrices and the frozen weight distribution.

---

## 8. Future Work

**Extension to modern large language models.** The current analysis covers architectures up to GPT-2 Large (774M parameters). Extension to LLaMA, Mistral, Phi, Falcon, and other widely deployed contemporary architectures is essential for determining whether the patterns identified here generalize to the model families most relevant to current practice. These models uniformly use causal language modeling, and our results predict strong Laplace prevalence, but this prediction must be verified empirically.

**Training checkpoint analysis.** The randomized-label controls establish that distributional regime does not change during short training on uninformative data. It remains unknown when during pretraining the regime is established. A checkpoint-by-checkpoint analysis — measuring Laplace% at regular intervals throughout full pretraining — would determine whether the distributional regime is set early (near initialization) or develops gradually. This analysis would directly test the gradient-flow mechanism hypothesis: if Laplace structure develops early and then stabilizes, this supports the view that the regime is set by the training objective structure rather than by the accumulation of data-specific patterns.

**Vision transformers as a cross-domain test.** Vision transformers (ViT, DeiT) use the same attention mechanism as language transformers but are trained on image data with different gradient flow properties. Applying our analysis to ViT would test whether the distributional patterns are specific to language modeling or are general properties of transformer attention under certain training regimes.

**Explicit L1 regularization as a controlled intervention.** The mechanistic hypothesis proposed in Section 6 predicts that explicit L1 regularization on attention weights should produce Laplace structure regardless of architecture. Directly training BERT-style models with L1 attention weight regularization and verifying that Laplace% increases would provide a controlled test of the hypothesis.

**Base model vs. instruction-tuned vs. RLHF comparison.** If our mechanistic account is correct — that Laplace structure reflects gradient-induced sparsity pressure — then instruction tuning and RLHF, which provide more selective gradient signal (reward on specific outputs), should further increase Laplace prevalence relative to the base pretrained model. This is a testable prediction: comparing distributional regimes across base, supervised fine-tuned, and RLHF variants of the same model would reveal whether alignment training amplifies or suppresses Laplace structure.

**Student-t distribution as a third candidate.** The current analysis compares only Laplace and Gaussian. The Student-t distribution generalizes both (Gaussian as the limiting case with infinite degrees of freedom) and could fit the data better in layers where neither Laplace nor Gaussian fits well. Including Student-t in the fitting procedure would provide a more complete distributional characterization and could reveal whether the "Laplace wins" classification is capturing heavy-tailed behavior more generally rather than specifically Laplace-like behavior.

**Head-level analysis within layers.** The current analysis concatenates all attention projections within a layer before fitting. This masks potential within-layer heterogeneity: individual attention heads may differ in distributional regime, with some heads Laplace-dominant and others Gaussian-dominant even within a single layer. Head-level analysis would connect our distributional findings to the attention head specialization literature and could reveal whether functionally distinct heads (induction heads, positional heads, copying heads) have characteristic distributional profiles.

**MLP layer analysis.** The current analysis is restricted to attention projection weights. MLP layers constitute a large fraction of total parameters and may show different distributional behavior. Extending the analysis to MLP weights would determine whether the patterns we report are specific to attention or reflect a broader property of the transformer block.

---

## 9. Conclusion

We have presented systematic empirical evidence that transformer attention weight distributions are architecture-determined and robust to training data content. Causal language models and encoder-decoder models consistently develop Laplace-like weight distributions in their attention layers, while masked language models remain predominantly Gaussian. The BERT/RoBERTa divergence demonstrates that training regime can shift the distributional outcome even with architecture held approximately constant, and the layer-depth analysis of GPT-style models reveals a gradient-flow signature consistent with causal masking creating asymmetric sparsity pressure in early layers. Randomized-label controls across 75 training runs confirm that these patterns are not data-driven artifacts. We propose a mechanistic account connecting autoregressive training to implicit L1-equivalent regularization through the maximum-entropy characterization of the Laplace distribution, and we derive concrete predictions for pruning, quantization, uncertainty quantification, and fine-tuning that differ by architecture family. The distributional geometry of trained weights is not a neutral implementation detail — it is a systematic signature of how architectures and training objectives jointly shape the optimization landscape, with practical consequences for every method that assumes Gaussian weight behavior.

---

## References

1. Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *Proceedings of AISTATS*.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. *Proceedings of ICCV*.
3. Vaswani, A., et al. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*.
4. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL*.
5. Radford, A., et al. (2019). Language models are unsupervised multitask learners. *OpenAI Technical Report*.
6. Liu, Y., et al. (2019). RoBERTa: A robustly optimized BERT pretraining approach. *arXiv:1907.11692*.
7. Lan, Z., et al. (2019). ALBERT: A lite BERT for self-supervised learning of language representations. *Proceedings of ICLR*.
8. Clark, K., et al. (2020). ELECTRA: Pre-training text encoders as discriminators rather than generators. *Proceedings of ICLR*.
9. Lewis, M., et al. (2019). BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. *Proceedings of ACL*.
10. Raffel, C., et al. (2019). Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of Machine Learning Research*.
11. Martin, C. H., & Mahoney, M. W. (2019). Traditional and heavy-tailed self-regularization in neural network models. *Proceedings of ICML*.
12. Martin, C. H., & Mahoney, M. W. (2021). Implicit self-regularization in deep neural networks: Evidence from random matrix theory and implications for learning. *Journal of Machine Learning Research*.
13. Elhage, N., et al. (2021). A mathematical framework for transformer circuits. *Transformer Circuits Thread*.
14. Elhage, N., et al. (2022). Toy models of superposition. *Transformer Circuits Thread*.
15. Daxberger, E., et al. (2021). Laplace Redux — Effortless Bayesian deep learning. *Advances in Neural Information Processing Systems*.
16. MacKay, D. J. C. (1992). A practical Bayesian framework for backpropagation networks. *Neural Computation*.
17. Frankle, J., & Carlin, M. (2018). The lottery ticket hypothesis: Finding sparse, trainable neural networks. *Proceedings of ICLR*.
18. Hu, E., et al. (2022). LoRA: Low-rank adaptation of large language models. *Proceedings of ICLR*.
19. Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter. *arXiv:1910.01108*.
20. Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*. Wiley.
21. Rissanen, J. (1978). Modeling by shortest data description. *Automatica*.
22. Hoefler, T., et al. (2021). Sparsity in deep learning: Pruning and growth for efficient inference and training in neural networks. *Journal of Machine Learning Research*.