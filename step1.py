import torch
from transformers import GPT2Model
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import laplace, norm
import numpy as np

model = GPT2Model.from_pretrained("gpt2")

first_layer = model.h[0].attn

weights = first_layer.c_attn.weight.detach().numpy()

print("Shape of the attention weights:", weights.shape)
print("First Values:", weights[0][:10])

plt.figure(figsize=(12, 6))
plt.imshow(weights[:50, :50], cmap='RdBu', aspect='auto')
plt.colorbar()
plt.title('Weights of the first attention layer of GPT-2 (50x50)')
plt.savefig('attention_weights.png')
print("Image saved as 'attention_weights.png'")

plt.figure(figsize=(8, 4))
plt.hist(weights.flatten(), bins=100, color='steelblue', edgecolor='none')
plt.title('Distribution of All Weights')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.savefig('weights_distribution.png')
print("Image saved as 'weights_distribution.png'")

weights_flat = weights.flatten()

x = np.linspace(weights_flat.min(), weights_flat.max(), 1000)

loc_l, scale_l = laplace.fit(weights_flat)
loc_n, scale_n = norm.fit(weights_flat)

plt.figure(figsize=(10, 5))
plt.hist(weights_flat, bins=100, density=True, color='steelblue', alpha=0.6, label='Weights real')
plt.plot(x, laplace.pdf(x, loc_l, scale_l), 'r-', linewidth=2, label='Fit Laplace')
plt.plot(x, norm.pdf(x, loc_n, scale_n), 'g--', linewidth=2, label='Fit Gaussian')
plt.legend()
plt.title('Weights GPT-2 vs Laplace vs Gaussian')
plt.savefig('distribution_comparison.png')
print("Third image saved")