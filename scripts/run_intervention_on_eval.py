#!/usr/bin/env python3
"""Run SAE interventions (ablation/clamping) on evaluation datasets.

Supports:
- ablate: Suppress language-specific features (projection removal)
- clamp: Steer toward language features (norm-scaled addition)

Usage:
    python scripts/run_intervention_on_eval.py \
        --intervention ablate \
        --language French \
        --strengths 0.0 3.0 6.0 \
        --eval_path data/dolly_200_prompts.json \
        --n_samples 50

    python scripts/run_intervention_on_eval.py \
        --intervention clamp \
        --language French \
        --strengths 0.0 0.1 0.2 \
        --eval_path data/french_comprehension_200.json
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import fire
import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from lib.core import (
    load_model_and_sae,
    get_mean_sae_activations,
    make_suppression_hook,
    make_steering_hook,
    generate,
)

torch.set_grad_enabled(False)

REPO_ROOT = Path(__file__).parent.parent
print("repo root:", REPO_ROOT)
RESULTS_DIR = REPO_ROOT / "results"
DATA_DIR = REPO_ROOT / "data"
LANGUAGES = ["French", "English", "German", "Dutch", "Italian"]


def find_language_features(model, tokenizer, sae, layer, language, n_sentences=100, keep_k=1):
    """Find top-k language-specific features using differential activation on FLORES."""
    with open(DATA_DIR / "flores_5lang_N1000.json") as f:
        flores = json.load(f)[:n_sentences]

    mean_acts = {}
    for lang in LANGUAGES:
        texts = [entry[lang] for entry in flores]
        mean_acts[lang] = get_mean_sae_activations(model, tokenizer, sae, layer, texts)

    # Compute specificity: target - max(others)
    target = mean_acts[language]
    other_max = torch.stack([mean_acts[l] for l in LANGUAGES if l != language]).max(dim=0).values
    spec = target - other_max
    spec[target <= 0] = -float("inf")

    # only keep top-k (keep_k) features
    top_vals, top_idxs = spec.topk(int(keep_k))
    features = [idx.item() for idx in top_idxs]

    return features


def load_eval(eval_path, n_samples=None):
    """Load evaluation data. Returns list of dicts with 'prompt' key."""
    with open(eval_path) as f:
        data = json.load(f)
    if n_samples:
        data = data[:n_samples]
    return data


def run_eval(model, tokenizer, prompts, hook_fn, layer, max_new_tokens):
    """Run generation on prompts with optional hook. Returns list of generations."""
    results = []
    for item in prompts:
        prompt = item["prompt"]
        output = generate(model, tokenizer, prompt, hook_fn=hook_fn, layer=layer, max_new_tokens=max_new_tokens)
        results.append({
            "id": item.get("id"),
            "prompt": prompt,
            "generation": output,
        })
    return results


def extract_letter(response: str) -> str:
    """Extract A/B/C/D from model response."""
    response = response.strip().upper()
    if response and response[0] in "ABCD":
        return response[0]
    match = re.search(r'\b([ABCD])\b', response)
    return match.group(1) if match else ""


def evaluate_mmlu(model, tokenizer, hook_fn, layer, n_samples: int = 100):
    """Evaluate on MMLU (all subjects) with optional intervention hook."""
    ds = load_dataset("cais/mmlu", "all", split=f"test[:{n_samples}]", trust_remote_code=True)

    history = []
    correct = 0
    for sample in tqdm(ds, desc="MMLU"):
        question = sample["question"]
        choices = sample["choices"]
        label = int(sample["answer"])

        options = "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(choices))
        prompt = f"{question}\n\n{options}\n\nAnswer with just the letter (A, B, C, or D):"

        response = generate(model, tokenizer, prompt, hook_fn=hook_fn, layer=layer, max_new_tokens=16)
        pred_letter = extract_letter(response)

        expected_letter = chr(65 + label)
        is_correct = pred_letter == expected_letter
        if is_correct:
            correct += 1

        history.append({
            "subject": sample.get("subject"),
            "question": question,
            "response": response,
            "predicted": pred_letter,
            "expected": expected_letter,
            "correct": is_correct,
        })

    return correct / len(ds), history


def main(
    intervention: str = "ablate",
    language: str = "French",
    strengths: list = None,
    eval_path: str = "dolly_200_prompts.json",
    n_samples: int = None,
    keep_k: int = 3,
    max_new_tokens: int = 128,
    model_name: str = "google/gemma-3-4b-it",
    sae_release: str = "gemma-scope-2-4b-pt-res",
    sae_id: str = "layer_29_width_65k_l0_medium",
    layer: int = 29,
    n_feature_sentences: int = 100,
    save_dir: str = None,
    run_mmlu: bool = False,
    mmlu_samples: int = 100,
):
    """
    Run intervention sweep on an evaluation dataset.

    Args:
        intervention: 'ablate' (suppress features) or 'clamp' (steer toward features)
        language: Target language for feature finding (French, English, German, Dutch, Italian)
        strengths: List of intervention strengths. For ablate: 0-10; for clamp: 0-0.3
        eval_path: Path to eval JSON (must have 'prompt' field per entry)
        n_samples: Number of eval samples (None = all)
        keep_k: Number of top language features to use
        max_new_tokens: Max tokens to generate
        model_name: HuggingFace model ID
        sae_release: SAELens release name
        sae_id: SAE ID within release
        layer: Layer to apply intervention
        n_feature_sentences: Number of FLORES sentences for feature finding
        save_dir: Output directory (default: results/)
    """
    # Defaults
    if strengths is None:
        strengths = [0.0, 3.0, 6.0] if intervention == "ablate" else [0.0, 0.1, 0.2]

    # Parse strengths from CLI (fire passes strings/lists inconsistently)
    # Supports: --strengths "0.0,3.0,6.0" or --strengths "[0.0,3.0,6.0]"
    if isinstance(strengths, str):
        strengths = strengths.strip("[]")
        strengths = [float(s.strip()) for s in strengths.split(",")]
    elif isinstance(strengths, (int, float)):
        strengths = [float(strengths)]
    else:
        strengths = [float(s) for s in strengths]

    # Validate
    assert intervention in ["ablate", "clamp"], f"Unknown intervention: {intervention}"
    assert language in LANGUAGES, f"Unknown language: {language}"

    # Config
    config = {
        "intervention": intervention,
        "language": language,
        "strengths": strengths,
        "eval_path": eval_path,
        "n_samples": n_samples,
        "keep_k": keep_k,
        "max_new_tokens": max_new_tokens,
        "model_name": model_name,
        "sae_release": sae_release,
        "sae_id": sae_id,
        "layer": layer,
        "n_feature_sentences": n_feature_sentences,
    }

    print("=" * 60)
    print("Intervention Evaluation")
    print("=" * 60)
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # Load model + SAE
    print("\nLoading model + SAE...")
    model, tokenizer, sae = load_model_and_sae(model_name, sae_release, sae_id)

    # Find language features
    print(f"\nFinding {language} features (layer {layer}, {n_feature_sentences} sentences)...")
    features = find_language_features(
        model, tokenizer, sae, layer, language,
        n_sentences=n_feature_sentences, keep_k=keep_k
    )
    config["features"] = features
    print(f"  Using features: {features}")

    # Load eval
    print(f"\nLoading eval: {eval_path}")
    eval_data = load_eval(DATA_DIR / eval_path if not Path(eval_path).is_absolute() else eval_path, n_samples)
    print(f"  {len(eval_data)} prompts")

    # Run sweep
    results = []
    for strength in strengths:
        print(f"\n{'='*40}")
        print(f"Strength: {strength}")
        print(f"{'='*40}")

        if strength == 0.0:
            hook_fn = None
        elif intervention == "ablate":
            hook_fn = make_suppression_hook(sae, features, strength=strength)
        else:  # clamp
            hook_fn = make_steering_hook(sae, features, coeff=strength)

        generations = run_eval(model, tokenizer, eval_data, hook_fn, layer, max_new_tokens)

        result_entry = {
            "strength": strength,
            "generations": generations,
        }

        # Run MMLU if requested
        if run_mmlu:
            print(f"  Running MMLU ({mmlu_samples} samples)...")
            mmlu_acc, mmlu_history = evaluate_mmlu(model, tokenizer, hook_fn, layer, n_samples=mmlu_samples)
            result_entry["mmlu_accuracy"] = mmlu_acc
            result_entry["mmlu_history"] = mmlu_history
            print(f"  MMLU accuracy: {mmlu_acc:.1%}")

        results.append(result_entry)
        print(f"  Generated {len(generations)} responses")

    # Save
    results_dir = RESULTS_DIR / save_dir if save_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_name = Path(eval_path).parent
    output_path = results_dir / f"{intervention}_{language}_{eval_name}_{keep_k}.json"

    output_data = {
        "config": config,
        "timestamp": timestamp,
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")
    print(f"{'='*60}")

    return output_path


if __name__ == "__main__":
    fire.Fire(main)

