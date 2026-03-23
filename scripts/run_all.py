"""Full language feature experiment: find features, test suppression, test steering.

Usage:
    conda activate severin
    python scripts/run_all.py                              # defaults: 4B, layer 29, 65k SAE
    python scripts/run_all.py --n_sentences 50             # faster (fewer FLORES sentences)
    python scripts/run_all.py --layer 22 --sae_id layer_22_width_65k_l0_medium  # different config

Key lessons baked in:
- Layer 29 (~85% depth in 4B's 34 layers) is where language routing features live.
  Layer 22 (~65%) has weaker, less monolingual features. Always match ~85% depth
  (e.g. layer 22 for 1B's 26 layers, layer 29 for 4B's 34 layers).
- PT SAEs (trained on base model) work fine on IT models — same features emerge.
- Suppression uses projection removal (subtract feat * decoder_dir from residual).
  Strength 3-6 cleanly kills French production while preserving comprehension.
- Steering uses norm-scaled decoder directions (Google's Gemma Scope 2 approach).
  Without norm scaling, the perturbation is negligible. coeff=0.1-0.2 is the sweet spot.
- Steering can also be applied ~8 layers before the SAE layer for stronger effect.

Expected results:
1. Feature finding: multiple features per language with >100x monolinguality ratio.
2. Suppression: "Write in French" → English fallback. "Translate French→English" → still works.
   German/English controls unaffected.
3. Steering: "What is the capital of Germany?" → answered in French at coeff=0.1-0.2.

Runtime: ~10 min on 1 GPU (mostly feature finding). Requires ~16GB VRAM for 4B model.
"""

import json
import sys
import argparse
sys.path.insert(0, ".")

import torch
torch.set_grad_enabled(False)

from lib.core import (
    load_model_and_sae, get_mean_sae_activations, generate,
    make_suppression_hook, make_steering_hook,
)

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument("--model_id", default="google/gemma-3-4b-it")
parser.add_argument("--sae_release", default="gemma-scope-2-4b-pt-res")
parser.add_argument("--sae_id", default="layer_29_width_65k_l0_medium")
parser.add_argument("--layer", type=int, default=29)
parser.add_argument("--n_sentences", type=int, default=100)
parser.add_argument("--data_path", default="data/flores_5lang_N1000.json")
args = parser.parse_args()

LANGUAGES = ["French", "English", "German", "Dutch", "Italian"]

# --- Load ---
print("Loading model + SAE...")
model, tokenizer, sae = load_model_and_sae(args.model_id, args.sae_release, args.sae_id)

with open(args.data_path) as f:
    flores = json.load(f)[:args.n_sentences]

# ============================================================
# 1. FIND LANGUAGE-SPECIFIC FEATURES
# ============================================================
print(f"\n{'='*70}")
print(f"FINDING LANGUAGE FEATURES | layer={args.layer} | {args.n_sentences} sentences")
print(f"{'='*70}")

mean_acts = {}
for lang in LANGUAGES:
    texts = [entry[lang] for entry in flores]
    mean_acts[lang] = get_mean_sae_activations(model, tokenizer, sae, args.layer, texts)
    print(f"  {lang}: {(mean_acts[lang] > 0).sum().item()} active features")

# Compute per-language specificity and select clean monolingual features
lang_features = {}
for lang in LANGUAGES:
    target = mean_acts[lang]
    other_max = torch.stack([mean_acts[l] for l in LANGUAGES if l != lang]).max(dim=0).values
    spec = target - other_max
    spec[target <= 0] = -float("inf")
    top_vals, top_idxs = spec.topk(10)

    print(f"\n  {lang}:")
    clean = []
    for idx, val in zip(top_idxs, top_vals):
        i = idx.item()
        ratio = target[i].item() / max(other_max[i].item(), 0.01)
        tag = " ***" if ratio > 100 else " **" if ratio > 10 else ""
        print(f"    {i:>8} | act={target[i]:>7.1f} | others={other_max[i]:>6.1f} | {ratio:>7.0f}x{tag}")
        if ratio > 3.0:
            clean.append(i)
    lang_features[lang] = clean

# Save
output = {"model": args.model_id, "sae": f"{args.sae_release}/{args.sae_id}",
          "layer": args.layer, "features": {l: lang_features[l] for l in LANGUAGES}}
with open("data/language_features_4b.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved features to data/language_features_4b.json")

# ============================================================
# 2. SUPPRESSION: PRODUCTION VS COMPREHENSION
# ============================================================
fr_feats = lang_features["French"][:7]
fr_text = flores[min(10, len(flores) - 1)]["French"][:150]
print(f"\n{'='*70}")
print(f"SUPPRESSION | French features={fr_feats}")
print(f"{'='*70}")

suppress_prompts = [
    ("PRODUCE French", "Write a short paragraph in French about the weather."),
    ("COMPREHEND French", f"Translate to English: '{fr_text}'"),
    ("CONTROL German", "Please say a sentence in German."),
    ("CONTROL English", "What is the capital of France?"),
]

for label, prompt in suppress_prompts:
    baseline = generate(model, tokenizer, prompt, layer=args.layer)
    print(f"\n  [{label}]")
    print(f"    baseline: {baseline[:140]}")
    for s in [3, 6]:
        hook = make_suppression_hook(sae, fr_feats, strength=s)
        out = generate(model, tokenizer, prompt, hook_fn=hook, layer=args.layer)
        print(f"    s={s}:      {out[:140]}")

# ============================================================
# 3. STEERING: FORCE FRENCH OUTPUT ON ENGLISH PROMPTS
# ============================================================
top_feat = lang_features["French"][0] if lang_features["French"] else None
print(f"\n{'='*70}")
print(f"STEERING | French feature={top_feat} | norm-scaled, KV-cache aware")
print(f"{'='*70}")

steer_prompts = [
    ("Capital", "What is the capital of Germany?"),
    ("Science", "Explain briefly why the sky is blue."),
    ("Fun fact", "Tell me a fun fact."),
]

if top_feat is not None:
    for label, prompt in steer_prompts:
        baseline = generate(model, tokenizer, prompt, layer=args.layer)
        print(f"\n  [{label}]")
        print(f"    baseline:   {baseline[:140]}")
        for c in [0.05, 0.1, 0.2]:
            hook = make_steering_hook(sae, top_feat, coeff=c)
            out = generate(model, tokenizer, prompt, hook_fn=hook, layer=args.layer)
            print(f"    coeff={c}: {out[:140]}")

    # Also try steering 8 layers before SAE (Google's approach — stronger effect)
    steer_early = max(0, args.layer - 8)
    print(f"\n  (steering at layer {steer_early} — 8 layers before SAE, stronger effect)")
    for label, prompt in steer_prompts:
        for c in [0.05, 0.1]:
            hook = make_steering_hook(sae, top_feat, coeff=c)
            out = generate(model, tokenizer, prompt, hook_fn=hook, layer=steer_early)
            print(f"    [{label}] coeff={c} L{steer_early}: {out[:140]}")
else:
    print("  No clean French features found, skipping steering.")

print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
