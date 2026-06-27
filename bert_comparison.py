import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import laplace, norm
from transformers import BertModel

# Load model
model = BertModel.from_pretrained("bert-base-uncased")

# Number of attention layers in BERT
num_layers = len(model.encoder.layer)
print(f"Analyzing {num_layers} BERT attention layers...\n")

# Store results for each layer
results = []

for layer_idx in range(num_layers):
    # BERT structure: model.encoder.layer[i].attention.self.query.weight
    weights_q = model.encoder.layer[layer_idx].attention.self.query.weight.detach().numpy()
    weights_k = model.encoder.layer[layer_idx].attention.self.key.weight.detach().numpy()
    weights_v = model.encoder.layer[layer_idx].attention.self.value.weight.detach().numpy()
    
    # Combine all attention projection weights
    weights = np.concatenate([weights_q.flatten(), weights_k.flatten(), weights_v.flatten()])
    
    # Fit distributions
    loc_l, scale_l = laplace.fit(weights)
    loc_n, scale_n = norm.fit(weights)
    
    # Calculate log-likelihood
    pdf_laplace = laplace.pdf(weights, loc_l, scale_l)
    pdf_gaussian = norm.pdf(weights, loc_n, scale_n)
    
    ll_laplace = np.sum(np.log(pdf_laplace + 1e-10))
    ll_gaussian = np.sum(np.log(pdf_gaussian + 1e-10))
    
    results.append({
        'layer': layer_idx,
        'laplace_loc': loc_l,
        'laplace_scale': scale_l,
        'gaussian_loc': loc_n,
        'gaussian_scale': scale_n,
        'll_laplace': ll_laplace,
        'll_gaussian': ll_gaussian,
        'better_fit': 'Laplace' if ll_laplace > ll_gaussian else 'Gaussian',
        'diff': abs(ll_laplace - ll_gaussian)
    })
    
    print(f"Layer {layer_idx}:")
    print(f"  Laplace LL: {ll_laplace:.2f}")
    print(f"  Gaussian LL: {ll_gaussian:.2f}")
    print(f"  Better fit: {results[-1]['better_fit']}")
    print()

# Visualization: Compare with GPT-2 pattern
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

layers = [r['layer'] for r in results]
better_fits = [r['better_fit'] for r in results]
diffs = [r['diff'] for r in results]
colors = ['red' if fit == 'Laplace' else 'green' for fit in better_fits]

# Plot 1: BERT layer-wise fit
ax = axes[0]
ax.bar(layers, diffs, color=colors, alpha=0.7)
ax.set_xlabel('Layer')
ax.set_ylabel('|LL Laplace - LL Gaussian|')
ax.set_title('BERT: Fit Quality Difference\n(Red=Laplace, Green=Gaussian)')
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Win count comparison (GPT-2 vs BERT)
ax = axes[1]
laplace_wins_bert = sum(1 for r in results if r['better_fit'] == 'Laplace')
gaussian_wins_bert = sum(1 for r in results if r['better_fit'] == 'Gaussian')
laplace_wins_gpt2 = 4
gaussian_wins_gpt2 = 8

x = np.arange(2)
width = 0.35

ax.bar(x - width/2, [laplace_wins_gpt2, laplace_wins_bert], width, label='Laplace', color='red', alpha=0.7)
ax.bar(x + width/2, [gaussian_wins_gpt2, gaussian_wins_bert], width, label='Gaussian', color='green', alpha=0.7)

ax.set_ylabel('Number of Layers')
ax.set_title('Distribution Fit Winner Count\n(GPT-2 vs BERT)')
ax.set_xticks(x)
ax.set_xticklabels(['GPT-2', 'BERT'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('gpt2_vs_bert_comparison.png', dpi=150)
print("Visualization saved as 'gpt2_vs_bert_comparison.png'")

# Summary
print("\n" + "="*60)
print("BERT SUMMARY")
print("="*60)
print(f"Laplace wins: {laplace_wins_bert}/{num_layers}")
print(f"Gaussian wins: {gaussian_wins_bert}/{num_layers}")
print(f"\nComparison with GPT-2:")
print(f"  GPT-2: Laplace {laplace_wins_gpt2}/12, Gaussian {gaussian_wins_gpt2}/12")
print(f"  BERT:  Laplace {laplace_wins_bert}/{num_layers}, Gaussian {gaussian_wins_bert}/{num_layers}")

# Check for similar pattern
gpt2_pattern = "Laplace-heavy early, Gaussian-heavy late"
bert_pattern = "Mixed" if (laplace_wins_bert > 3 and laplace_wins_bert < 9) else ("Laplace-dominant" if laplace_wins_bert > 9 else "Gaussian-dominant")
print(f"\nPattern: {bert_pattern}")
