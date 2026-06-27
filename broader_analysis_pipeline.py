import json
import os
import re
from typing import Dict, List

import numpy as np
import torch
from scipy.stats import laplace, norm, kurtosis
from transformers import AutoConfig, AutoModel


DEFAULT_MODEL_IDS = [
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

OPTIONAL_MODEL_IDS = [
    "meta-llama/Llama-3.2-1B",
    "EleutherAI/gpt-j-6b",
    "microsoft/phi-2",
    "google/vit-base-patch16-224",
]

MAX_LAYERS_PER_MODEL = 8

ATTENTION_TOKENS = [
    "attention",
    "selfattention",
    "attn",
    "q_proj",
    "k_proj",
    "v_proj",
    "query",
    "key",
    "value",
    "c_attn",
    "in_proj_weight",
    "q_lin",
    "k_lin",
    "v_lin",
]


def collect_attention_tensors(model: torch.nn.Module) -> List[np.ndarray]:
    """Collect one attention-weight vector per layer, up to MAX_LAYERS_PER_MODEL."""
    layer_groups: Dict[int, List[np.ndarray]] = {}
    layer_pattern = re.compile(r"(?:^|\.)(h|layer|layers|block|albert_layers)\.(\d+)")

    for name, param in model.named_parameters():
        if not name.endswith(".weight") or param.ndim < 2:
            continue
        lowered = name.lower()
        if not any(token in lowered for token in ATTENTION_TOKENS):
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


def summarize_layerwise_fit(tensors: List[np.ndarray]) -> Dict[str, object]:
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
        "num_layers": len(results),
        "laplace_wins": laplace_wins,
        "gaussian_wins": gaussian_wins,
        "laplace_pct": 100.0 * laplace_wins / max(1, len(results)),
        "layers": results,
    }


def summarize_initialization(model: torch.nn.Module) -> Dict[str, float]:
    tensors = collect_attention_tensors(model)
    flat = np.concatenate(tensors).astype(np.float64).reshape(-1)
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "range": float(np.max(flat) - np.min(flat)),
        "kurtosis": float(kurtosis(flat)),
        "skewness": float(np.mean((flat - np.mean(flat)) ** 3) / (np.std(flat) ** 3)),
    }


def analyze_model(model_id: str) -> Dict[str, object]:
    config = AutoConfig.from_pretrained(model_id)
    pretrained_model = AutoModel.from_pretrained(model_id)
    random_model = AutoModel.from_config(config)

    pretrained_summary = summarize_layerwise_fit(collect_attention_tensors(pretrained_model))
    random_summary = summarize_layerwise_fit(collect_attention_tensors(random_model))
    init_stats = summarize_initialization(random_model)

    return {
        "model_id": model_id,
        "architecture": config.model_type,
        "pretrained": pretrained_summary,
        "random_init": random_summary,
        "init_stats": init_stats,
    }


def main() -> None:
    torch.set_grad_enabled(False)
    model_ids = DEFAULT_MODEL_IDS
    include_optional = os.environ.get("INCLUDE_OPTIONAL_MODELS", "0") == "1"
    if include_optional:
        model_ids = model_ids + OPTIONAL_MODEL_IDS

    print("=" * 96)
    print("BROADER ANALYSIS PIPELINE")
    print("=" * 96)
    print(f"Models to analyze: {len(model_ids)}")

    results = []
    failures = []

    for model_id in model_ids:
        try:
            print(f"\n[{model_id}]")
            result = analyze_model(model_id)
            results.append(result)
            print(f"  pretrained laplace %: {result['pretrained']['laplace_pct']:.1f}")
            print(f"  random-init laplace %: {result['random_init']['laplace_pct']:.1f}")
            print(f"  init kurtosis: {result['init_stats']['kurtosis']:.4f}")
        except Exception as exc:  # pragma: no cover - runtime robustness
            failures.append({"model_id": model_id, "error": str(exc)})
            print(f"  failed: {exc}")

    output_path = os.path.join(os.getcwd(), "broader_analysis_results.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump({"results": results, "failures": failures, "model_count": len(results)}, handle, indent=2)

    print("\n" + "=" * 96)
    print(f"Completed: {len(results)} successful analyses, {len(failures)} failures")
    print(f"Saved results to {output_path}")

    if results:
        print("\nTop pretrained Laplace percentages:")
        for item in sorted(results, key=lambda x: x["pretrained"]["laplace_pct"], reverse=True)[:10]:
            print(f"  {item['model_id']}: {item['pretrained']['laplace_pct']:.1f}%")


if __name__ == "__main__":
    main()
