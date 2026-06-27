import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import laplace, norm
from transformers import GPT2Model, BertModel

print("="*60)
print("TESTING UNTRAINED (RANDOM INIT) MODELS")
print("="*60)

# Function to analyze layers
def analyze_model(model, model_name, is_gpt2=True):
    if is_gpt2:
        num_layers = len(model.h)
        results = []
        for layer_idx in range(num_layers):
            weights = model.h[layer_idx].attn.c_attn.weight.detach().numpy()
            weights_flat = weights.flatten()
            
            loc_l, scale_l = laplace.fit(weights_flat)
            loc_n, scale_n = norm.fit(weights_flat)
            
            pdf_laplace = laplace.pdf(weights_flat, loc_l, scale_l)
            pdf_gaussian = norm.pdf(weights_flat, loc_n, scale_n)
            
            ll_laplace = np.sum(np.log(pdf_laplace + 1e-10))
            ll_gaussian = np.sum(np.log(pdf_gaussian + 1e-10))
            
            results.append({
                'layer': layer_idx,
                'll_laplace': ll_laplace,
                'll_gaussian': ll_gaussian,
                'better_fit': 'Laplace' if ll_laplace > ll_gaussian else 'Gaussian'
            })
    else:
        num_layers = len(model.encoder.layer)
        results = []
        for layer_idx in range(num_layers):
            weights_q = model.encoder.layer[layer_idx].attention.self.query.weight.detach().numpy()
            weights_k = model.encoder.layer[layer_idx].attention.self.key.weight.detach().numpy()
            weights_v = model.encoder.layer[layer_idx].attention.self.value.weight.detach().numpy()
            
            weights = np.concatenate([weights_q.flatten(), weights_k.flatten(), weights_v.flatten()])
            
            loc_l, scale_l = laplace.fit(weights)
            loc_n, scale_n = norm.fit(weights)
            
            pdf_laplace = laplace.pdf(weights, loc_l, scale_l)
            pdf_gaussian = norm.pdf(weights, loc_n, scale_n)
            
            ll_laplace = np.sum(np.log(pdf_laplace + 1e-10))
            ll_gaussian = np.sum(np.log(pdf_gaussian + 1e-10))
            
            results.append({
                'layer': layer_idx,
                'll_laplace': ll_laplace,
                'll_gaussian': ll_gaussian,
                'better_fit': 'Laplace' if ll_laplace > ll_gaussian else 'Gaussian'
            })
    
    laplace_wins = sum(1 for r in results if r['better_fit'] == 'Laplace')
    gaussian_wins = sum(1 for r in results if r['better_fit'] == 'Gaussian')
    
    print(f"\n{model_name}:")
    print(f"  Laplace wins: {laplace_wins}/{num_layers}")
    print(f"  Gaussian wins: {gaussian_wins}/{num_layers}")
    
    return results, laplace_wins, gaussian_wins

# Load UNTRAINED models (config only, random weights)
print("\nLoading untrained GPT-2...")
gpt2_config = torch.nn.Module()
from transformers import GPT2Config
gpt2_untrained = GPT2Model(GPT2Config())
gpt2_results, gpt2_l, gpt2_g = analyze_model(gpt2_untrained, "GPT-2 (UNTRAINED)", is_gpt2=True)

print("\nLoading untrained BERT...")
from transformers import BertConfig
bert_untrained = BertModel(BertConfig())
bert_results, bert_l, bert_g = analyze_model(bert_untrained, "BERT (UNTRAINED)", is_gpt2=False)

# Compare with pre-trained
print("\n" + "="*60)
print("LOADING PRE-TRAINED FOR COMPARISON")
print("="*60)

print("\nLoading pre-trained GPT-2...")
gpt2_pretrained = GPT2Model.from_pretrained("gpt2")
gpt2_pt_results, gpt2_pt_l, gpt2_pt_g = analyze_model(gpt2_pretrained, "GPT-2 (PRETRAINED)", is_gpt2=True)

print("\nLoading pre-trained BERT...")
bert_pretrained = BertModel.from_pretrained("bert-base-uncased")
bert_pt_results, bert_pt_l, bert_pt_g = analyze_model(bert_pretrained, "BERT (PRETRAINED)", is_gpt2=False)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

models = ['GPT-2\n(Untrained)', 'GPT-2\n(Pretrained)', 'BERT\n(Untrained)', 'BERT\n(Pretrained)']
laplace_wins_list = [gpt2_l, gpt2_pt_l, bert_l, bert_pt_l]
gaussian_wins_list = [gpt2_g, gpt2_pt_g, bert_g, bert_pt_g]

# Plot 1: Win counts
ax = axes[0, 0]
x = np.arange(len(models))
width = 0.35
ax.bar(x - width/2, laplace_wins_list, width, label='Laplace', color='red', alpha=0.7)
ax.bar(x + width/2, gaussian_wins_list, width, label='Gaussian', color='green', alpha=0.7)
ax.set_ylabel('Number of Layers')
ax.set_title('Distribution Fit Winner Count\n(Untrained vs Pretrained)')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim([0, 13])
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Laplace percentage
ax = axes[0, 1]
laplace_pcts = [l/12*100 for l in laplace_wins_list]
colors_bar = ['steelblue', 'navy', 'orange', 'darkorange']
ax.bar(models, laplace_pcts, color=colors_bar, alpha=0.7)
ax.set_ylabel('Laplace Win Percentage (%)')
ax.set_title('Laplace Dominance by Model')
ax.set_ylim([0, 100])
ax.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(laplace_pcts):
    ax.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')

# Plot 3: Layer-by-layer comparison GPT-2
ax = axes[1, 0]
layers = range(12)
ax.plot(layers, [r['better_fit'] == 'Laplace' for r in gpt2_results], 'o-', label='Untrained', linewidth=2, markersize=6)
ax.plot(layers, [r['better_fit'] == 'Laplace' for r in gpt2_pt_results], 's--', label='Pretrained', linewidth=2, markersize=6)
ax.set_xlabel('Layer')
ax.set_ylabel('Laplace Fit (1=Yes, 0=No)')
ax.set_title('GPT-2: Untrained vs Pretrained (layer-wise)')
ax.legend()
ax.set_ylim([-0.1, 1.1])
ax.grid(True, alpha=0.3)

# Plot 4: Layer-by-layer comparison BERT
ax = axes[1, 1]
ax.plot(layers, [r['better_fit'] == 'Laplace' for r in bert_results], 'o-', label='Untrained', linewidth=2, markersize=6)
ax.plot(layers, [r['better_fit'] == 'Laplace' for r in bert_pt_results], 's--', label='Pretrained', linewidth=2, markersize=6)
ax.set_xlabel('Layer')
ax.set_ylabel('Laplace Fit (1=Yes, 0=No)')
ax.set_title('BERT: Untrained vs Pretrained (layer-wise)')
ax.legend()
ax.set_ylim([-0.1, 1.1])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('untrained_vs_pretrained.png', dpi=150)
print("\n\nVisualization saved as 'untrained_vs_pretrained.png'")

# Summary
print("\n" + "="*60)
print("KEY FINDING")
print("="*60)
print(f"\nGPT-2 Untrained:   {gpt2_l} Laplace, {gpt2_g} Gaussian")
print(f"GPT-2 Pretrained:  {gpt2_pt_l} Laplace, {gpt2_pt_g} Gaussian")
print(f"→ Pretrained shifted TOWARDS Gaussian? {gpt2_pt_g > gpt2_g}")

print(f"\nBERT Untrained:    {bert_l} Laplace, {bert_g} Gaussian")
print(f"BERT Pretrained:   {bert_pt_l} Laplace, {bert_pt_g} Gaussian")
print(f"→ Pretrained shifted TOWARDS Gaussian? {bert_pt_g > bert_g}")

print("\nConclusion:")
if (gpt2_pt_g >= gpt2_g) and (bert_pt_g >= bert_g):
    print("✓ BOTH models: Training shifts weights toward Gaussian distribution")
    print("  → Distribution fit is LEARNED during pretraining, not baked into init")
else:
    print("✗ Mixed results: Training effect is more subtle")
