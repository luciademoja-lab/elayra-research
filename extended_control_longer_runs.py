import json
import os
import re
from typing import Dict, List

import numpy as np
import torch
from scipy.stats import laplace, norm
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer


CONTROL_MODELS = [
    ("gpt2", "causal"),
    ("bert-base-uncased", "masked"),
    ("facebook/electra-base-discriminator", "masked"),  # Intermediate case at 62.5%
]

MAX_LAYERS = 8
TRAINING_STEPS = 500
BATCH_SIZE = 16
SEED = 42


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


def build_batch(tokenizer, model_type: str, batch_size: int, seq_len: int = 32) -> Dict[str, torch.Tensor]:
    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def run_control_extended(model_id: str, model_type: str, seed: int) -> Dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    if model_type == "causal":
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    else:
        model = AutoModelForMaskedLM.from_pretrained(model_id).to(device)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    before = summarize_layerwise_fit(collect_attention_tensors(model))

    loss_history = []
    checkpoint_steps = [0, 50, 100, 250, 500]
    distributions_at_steps = {0: before}

    for step in range(TRAINING_STEPS):
        batch = build_batch(tokenizer, model_type, BATCH_SIZE)
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss_history.append(float(loss))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Capture distributional state at key checkpoints
        if step + 1 in checkpoint_steps and step + 1 != 0:
            try:
                dist = summarize_layerwise_fit(collect_attention_tensors(model))
                distributions_at_steps[step + 1] = dist
            except Exception:
                pass

    after = summarize_layerwise_fit(collect_attention_tensors(model))
    distributions_at_steps[TRAINING_STEPS] = after

    return {
        "seed": seed,
        "before": before,
        "after": after,
        "loss_history": loss_history,
        "distributions_at_steps": distributions_at_steps,
    }


def main() -> None:
    all_results = []

    print("=" * 96)
    print("EXTENDED RANDOMIZED-LABEL CONTROL: 500 TRAINING STEPS")
    print("=" * 96)
    print(f"Models: {len(CONTROL_MODELS)}")
    print(f"Seed: {SEED}")
    print(f"Training steps: {TRAINING_STEPS}")
    print(f"Batch size: {BATCH_SIZE}\n")

    for model_id, model_type in CONTROL_MODELS:
        try:
            print(f"[{model_id}] ({model_type})")
            result = run_control_extended(model_id, model_type, SEED)
            
            before_pct = result["before"]["laplace_pct"]
            after_pct = result["after"]["laplace_pct"]
            delta = after_pct - before_pct
            
            print(f"  Laplace %: {before_pct:.1f}% (step 0) → {after_pct:.1f}% (step {TRAINING_STEPS}) | Δ {delta:.2f}%")
            print(f"  Loss trajectory: {result['loss_history'][0]:.3f} → {result['loss_history'][-1]:.3f}")
            
            # Report intermediate steps
            print(f"  Distributional stability across training:")
            for step in sorted(result["distributions_at_steps"].keys()):
                pct = result["distributions_at_steps"][step]["laplace_pct"]
                print(f"    Step {step:3d}: {pct:.1f}% Laplace")
            
            all_results.append({
                "model_id": model_id,
                "model_type": model_type,
                "result": result,
            })
            print()
        except Exception as exc:
            print(f"  failed: {exc}\n")

    out_path = os.path.join(os.getcwd(), "extended_control_500steps.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({"results": all_results}, handle, indent=2)

    print("=" * 96)
    print(f"Saved to {out_path}\n")
    print("Summary:")
    for item in all_results:
        before = item["result"]["before"]["laplace_pct"]
        after = item["result"]["after"]["laplace_pct"]
        delta = after - before
        print(f"  {item['model_id']}: {before:.1f}% → {after:.1f}% (Δ {delta:.2f}%)")


if __name__ == "__main__":
    main()
