import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import laplace, norm
from transformers import GPT2Model

# Load model
model = GPT2Model.from_pretrained("gpt2")

# Number of attention layers in GPT-2
num_layers = len(model.h)
print(f"Analyzing {num_layers} attention layers...\n")

# Store results for each layer
results = []

for layer_idx in range(num_layers):
    weights = model.h[layer_idx].attn.c_attn.weight.detach().numpy()
    weights_flat = weights.flatten()
    
    # Fit distributions
    loc_l, scale_l = laplace.fit(weights_flat)
    loc_n, scale_n = norm.fit(weights_flat)
    
    # Calculate KL divergence or other metrics
    x = np.linspace(weights_flat.min(), weights_flat.max(), 1000)
    
    # Simple metric: compare PDF values at quantiles
    pdf_laplace = laplace.pdf(weights_flat, loc_l, scale_l)
    pdf_gaussian = norm.pdf(weights_flat, loc_n, scale_n)
    
    # Log-likelihood
    ll_laplace = np.sum(np.log(pdf_laplace + 1e-10))
    ll_gaussian = np.sum(np.log(pdf_gaussian + 1e-10))
    
    results.append({
        'layer': layer_idx,
        'weights_shape': weights.shape,
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
    print(f"  Shape: {weights.shape}")
    print(f"  Laplace LL: {ll_laplace:.2f}")
    print(f"  Gaussian LL: {ll_gaussian:.2f}")
    print(f"  Better fit: {results[-1]['better_fit']}")
    print()

# Visualization 1: Log-likelihood comparison across layers
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

layers = [r['layer'] for r in results]
ll_laplace_vals = [r['ll_laplace'] for r in results]
ll_gaussian_vals = [r['ll_gaussian'] for r in results]
diffs = [r['diff'] for r in results]

# Plot 1: Log-likelihoods
ax = axes[0, 0]
ax.plot(layers, ll_laplace_vals, 'r-o', linewidth=2, label='Laplace', markersize=6)
ax.plot(layers, ll_gaussian_vals, 'g-s', linewidth=2, label='Gaussian', markersize=6)
ax.set_xlabel('Layer')
ax.set_ylabel('Log-Likelihood')
ax.set_title('Distribution Fit Quality Across Layers')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Difference (how much better is Laplace)
ax = axes[0, 1]
colors = ['red' if r['better_fit'] == 'Laplace' else 'green' for r in results]
ax.bar(layers, diffs, color=colors, alpha=0.7)
ax.set_xlabel('Layer')
ax.set_ylabel('|LL Laplace - LL Gaussian|')
ax.set_title('Fit Quality Difference (Laplace vs Gaussian)')
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Scale parameters
ax = axes[1, 0]
laplace_scales = [r['laplace_scale'] for r in results]
gaussian_scales = [r['gaussian_scale'] for r in results]
ax.plot(layers, laplace_scales, 'r-o', linewidth=2, label='Laplace scale', markersize=6)
ax.plot(layers, gaussian_scales, 'g-s', linewidth=2, label='Gaussian σ', markersize=6)
ax.set_xlabel('Layer')
ax.set_ylabel('Scale Parameter')
ax.set_title('Scale Parameter Across Layers')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Win count
ax = axes[1, 1]
laplace_wins = sum(1 for r in results if r['better_fit'] == 'Laplace')
gaussian_wins = sum(1 for r in results if r['better_fit'] == 'Gaussian')
ax.bar(['Laplace', 'Gaussian'], [laplace_wins, gaussian_wins], color=['red', 'green'], alpha=0.7)
ax.set_ylabel('Number of Layers')
ax.set_title(f'Distribution Fit Winner Count (Total: {num_layers} layers)')
for i, v in enumerate([laplace_wins, gaussian_wins]):
    ax.text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('layer_comparison.png', dpi=150)
print("Visualization saved as 'layer_comparison.png'")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Laplace wins: {laplace_wins}/{num_layers}")
print(f"Gaussian wins: {gaussian_wins}/{num_layers}")
print(f"Unanimous conclusion: Laplace is universally better" if laplace_wins == num_layers else f"Mixed results")
