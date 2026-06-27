import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.stats import laplace, norm
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer


CONTROL_MODELS = [
    ("gpt2", "causal"),
    ("bert-base-uncased", "masked"),
]

MAX_LAYERS = 8


def collect_attention_tensors(model: torch.nn.Module) -> List[np.ndarray]:
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
        raise ValueError("No attention weights found")

    tensors = []
    for layer_idx in sorted(layer_groups)[:MAX_LAYERS]:
        tensors.append(np.concatenate(layer_groups[layer_idx]).astype(np.float64))
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


def build_batch(tokenizer, model_type: str, batch_size: int = 4, seq_len: int = 16) -> Dict[str, torch.Tensor]:
    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    if model_type == "causal":
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    else:
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def run_control(model_id: str, model_type: str) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if model_type == "causal":
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    else:
        model = AutoModelForMaskedLM.from_pretrained(model_id).to(device)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    batch = build_batch(tokenizer, model_type)
    batch = {k: v.to(device) for k, v in batch.items()}

    before = summarize_layerwise_fit(collect_attention_tensors(model))

    for _ in range(10):
        batch = build_batch(tokenizer, model_type)
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    after = summarize_layerwise_fit(collect_attention_tensors(model))

    return {
        "model_id": model_id,
        "model_type": model_type,
        "before": before,
        "after": after,
        "loss": float(loss.item()) if 'loss' in locals() else None,
    }


def main() -> None:
    torch.manual_seed(0)
    results = []
    for model_id, model_type in CONTROL_MODELS:
        try:
            print(f"\nRunning shuffled-label control for {model_id}...")
            result = run_control(model_id, model_type)
            results.append(result)
            print(f"  before: {result['before']['laplace_pct']:.1f}% Laplace")
            print(f"  after: {result['after']['laplace_pct']:.1f}% Laplace")
        except Exception as exc:
            print(f"  failed: {exc}")

    out_path = os.path.join(os.getcwd(), "shuffled_control_results.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({"results": results}, handle, indent=2)

    print(f"\nSaved control results to {out_path}")


if __name__ == "__main__":
    main()
