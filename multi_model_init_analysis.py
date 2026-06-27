import json
import os
from typing import List, Dict, Tuple

import numpy as np
import torch
from scipy.stats import kurtosis
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


def find_attention_weight_tensor(model: torch.nn.Module) -> np.ndarray:
    """Return a flattened tensor from an attention projection weight.

    The selection is heuristic and based on parameter names so it works across
    several transformer families without hard-coding each architecture.
    """
    candidates = []
    for name, param in model.named_parameters():
        lowered = name.lower()
        if param.ndim < 2:
            continue
        if not lowered.endswith(".weight"):
            continue
        if any(token in lowered for token in ["c_attn", "q_proj", "k_proj", "v_proj", "query", "key", "value", "in_proj_weight", "q_lin", "k_lin", "v_lin", "selfattention"]):
            candidates.append((name, param))

    if not candidates:
        raise ValueError("No attention projection weights were found")

    preferred = [c for c in candidates if any(token in c[0].lower() for token in ["c_attn", "in_proj_weight", "q_proj", "query", "selfattention.q"])]
    if preferred:
        candidates = preferred

    tensor = candidates[0][1].detach().cpu().numpy().reshape(-1)
    return tensor


def summarize_model(model_id: str) -> Dict[str, object]:
    config = AutoConfig.from_pretrained(model_id)
    model = AutoModel.from_config(config)
    weights = find_attention_weight_tensor(model)

    return {
        "model_id": model_id,
        "architecture": config.model_type,
        "num_params": sum(p.numel() for p in model.parameters()),
        "weight_mean": float(np.mean(weights)),
        "weight_std": float(np.std(weights)),
        "weight_range": float(np.max(weights) - np.min(weights)),
        "weight_kurtosis": float(kurtosis(weights)),
        "weight_skewness": float(np.mean((weights - np.mean(weights)) ** 3) / (np.std(weights) ** 3)),
    }


def main() -> None:
    torch.set_grad_enabled(False)
    results = []
    failures = []

    print("=" * 72)
    print("EXPANDED INITIALIZATION ANALYSIS")
    print("=" * 72)
    print(f"Attempting to evaluate {len(MODEL_IDS)} model ids")

    for model_id in MODEL_IDS:
        try:
            print(f"\n[{model_id}] loading...")
            summary = summarize_model(model_id)
            results.append(summary)
            print(f"  architecture: {summary['architecture']}")
            print(f"  kurtosis: {summary['weight_kurtosis']:.4f}")
            print(f"  std: {summary['weight_std']:.6f}")
        except Exception as exc:  # pragma: no cover - runtime robustness
            failures.append({"model_id": model_id, "error": str(exc)})
            print(f"  failed: {exc}")

    output_path = os.path.join(os.getcwd(), "expanded_model_init_results.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump({"results": results, "failures": failures}, handle, indent=2)

    print("\n" + "=" * 72)
    print(f"Completed: {len(results)} successful model summaries, {len(failures)} failures")
    print(f"Saved results to {output_path}")

    if results:
        sorted_results = sorted(results, key=lambda item: item["weight_kurtosis"], reverse=True)
        print("\nTop kurtosis values:")
        for item in sorted_results[:10]:
            print(f"  {item['model_id']}: kurtosis={item['weight_kurtosis']:.4f}, std={item['weight_std']:.6f}")


if __name__ == "__main__":
    main()
