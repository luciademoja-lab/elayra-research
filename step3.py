import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2Model

model = GPT2Model.from_pretrained('gpt2')
weights = model.h[0].attn.c_attn.weight.detach().numpy()

row = weights[0]

fourier_ft = np.fft.fft(row)
frequencies = np.fft.fftfreq(len(row))
amplitude = np.abs(fourier_ft)

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(row)
plt.title('Pesi riga 0 - spazio originale')

plt.subplot(2, 1, 2)
plt.plot(frequencies[:len(frequencies)//2], amplitude[:len(amplitude)//2])
plt.title('Spettro di Fourier - frequenze')
plt.xlabel('Frequenza')
plt.ylabel('Ampiezza')

plt.tight_layout()
plt.savefig('fourier_pesi.png')
print("Immagine salvata")

dominant_freq = frequencies[np.argmax(amplitude[1:len(amplitude)//2])+1]
print(f"Dominant frequency: {dominant_freq:.4f}")
print(f"Max amplitude: {amplitude.max():.4f}")
print(f"Amplitude mean: {amplitude.mean():.4f}")