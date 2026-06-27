import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2Model

model = GPT2Model.from_pretrained('gpt2')
weights = model.h[0].attn.c_attn.weight.detach().numpy().flatten()

phi = (1 + np.sqrt(5)) / 2  # 1.6180...

weights_abs = np.sort(np.abs(weights))[::-1]

rapports = weights_abs[:-1] / (weights_abs[1:] + 1e-10)

tolerance = 0.05
close_to_phi = np.abs(rapports - phi) < tolerance
percentage = close_to_phi.sum() / len(rapports) * 100

print(f"Rapporti vicini a φ ({phi:.4f}): {percentage:.2f}%")
print(f"Media dei rapporti: {rapports.mean():.4f}")
print(f"Mediana dei rapporti: {np.median(rapports):.4f}")

plt.figure(figsize=(10, 5))
plt.hist(rapports, bins=200, range=(0, 5), color='steelblue', alpha=0.7)
plt.axvline(x=phi, color='red', linewidth=2, label=f'φ = {phi:.4f}')
plt.axvline(x=rapports.mean(), color='green', linewidth=2, linestyle='--', label=f'Media = {rapports.mean():.4f}')
plt.legend()
plt.title('Rapports between consecutive weights vs φ')
plt.savefig('rapports_phi.png')
print("Immage saved as 'rapports_phi.png'")