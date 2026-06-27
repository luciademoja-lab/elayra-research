import json
import os
import re
from typing import Dict, List

import numpy as np
import torch
from scipy.stats import laplace, norm
from transformers import AutoConfig, AutoModel


MODEL_IDS = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "bert-base-uncased",
    "bert-large-uncased",
    "distilbert-base-uncased",
    "roberta-base",
    "roberta-large",
    "distilroberta-base",
    "albert-base-v2",
    "google/electra-small-discriminator",
    "facebook/bart-base",
    "t5-small",
    "t5-base",
    "google/mt5-small",
]

MAX_LAYERS_PER_MODEL = 15


def get_attention_weight_tensors(model: torch.nn.Module) -> List[np.ndarray]:
    """Collect one flattened attention-weight vector per layer, up to MAX_LAYERS_PER_MODEL."""
    layer_groups: Dict[int, List[np.ndarray]] = {}
    layer_pattern = re.compile(r"(?:^|\.)(h|layer|layers|block|albert_layers)\.(\d+)")

    for name, param in model.named_parameters():
        if not name.endswith(".weight") or param.ndim < 2:
            continue
        lowered = name.lower()
        if not any(token in lowered for token in ["attention", "selfattention", "attn", "q_proj", "k_proj", "v_proj", "query", "key", "value", "c_attn", "in_proj_weight", "q_lin", "k_lin", "v_lin"]):
            continue

        match = layer_pattern.search(lowered)
        if not match:
            continue
        layer_idx = int(match.group(2))
        layer_groups.setdefault(layer_idx, []).append(param.detach().cpu().numpy().reshape(-1))

    if not layer_groups:
        raise ValueError("No attention weights found for this model")

    tensors = []
    for layer_idx in sorted(layer_groups)[:MAX_LAYERS_PER_MODEL]:
        tensors.append(np.concatenate(layer_groups[layer_idx]).astype(np.float64))

    if not tensors:
        raise ValueError("No layerwise attention weights were assembled")

    return tensors


def analyze_layerwise(model_id: str) -> Dict[str, object]:
    config = AutoConfig.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    tensors = get_attention_weight_tensors(model)

    results = []
    for idx, weights in enumerate(tensors):
        flat = weights.reshape(-1)
        loc_l, scale_l = laplace.fit(flat)
        loc_n, scale_n = norm.fit(flat)
        pdf_laplace = laplace.pdf(flat, loc_l, scale_l)
        pdf_gaussian = norm.pdf(flat, loc_n, scale_n)
        ll_laplace = float(np.sum(np.log(pdf_laplace + 1e-10)))
        ll_gaussian = float(np.sum(np.log(pdf_gaussian + 1e-10)))
        better_fit = "Laplace" if ll_laplace > ll_gaussian else "Gaussian"
        results.append({
            "layer": idx,
            "ll_laplace": ll_laplace,
            "ll_gaussian": ll_gaussian,
            "better_fit": better_fit,
        })

    laplace_wins = sum(1 for r in results if r["better_fit"] == "Laplace")
    gaussian_wins = sum(1 for r in results if r["better_fit"] == "Gaussian")

    return {
        "model_id": model_id,
        "architecture": config.model_type,
        "num_layers": len(results),
        "laplace_wins": laplace_wins,
        "gaussian_wins": gaussian_wins,
        "laplace_pct": 100.0 * laplace_wins / max(1, len(results)),
        "layers": results,
    }


def main() -> None:
    torch.set_grad_enabled(False)
    all_results = []
    failures = []

    print("=" * 90)
    print("LAYERWISE MODEL COMPARISON")
    print("=" * 90)

    for model_id in MODEL_IDS:
        try:
            print(f"\n[{model_id}] analyzing first {MAX_LAYERS_PER_MODEL} attention layers...")
            result = analyze_layerwise(model_id)
            all_results.append(result)
            print(f"  layers analyzed: {result['num_layers']}")
            print(f"  laplace wins: {result['laplace_wins']}")
            print(f"  gaussian wins: {result['gaussian_wins']}")
            print(f"  laplace %: {result['laplace_pct']:.1f}")
        except Exception as exc:
            failures.append({"model_id": model_id, "error": str(exc)})
            print(f"  failed: {exc}")

    out_path = os.path.join(os.getcwd(), "layerwise_model_comparison.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({"results": all_results, "failures": failures}, handle, indent=2)

    print("\n" + "=" * 90)
    print(f"Completed: {len(all_results)} successful model analyses, {len(failures)} failures")
    print(f"Saved results to {out_path}")

    if all_results:
        print("\nSummary by model:")
        for item in sorted(all_results, key=lambda x: x["laplace_pct"], reverse=True):
            print(f"  {item['model_id']}: {item['laplace_pct']:.1f}% Laplace ({item['laplace_wins']}/{item['num_layers']} layers)")


if __name__ == "__main__":
    main()
