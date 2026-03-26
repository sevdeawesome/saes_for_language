# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Publication-quality research on **language-specific SAE features** in multilingual LLMs. The core novel finding: **ablating language production features does not affect language comprehension** — suggesting SAEs capture output routing circuits, not unified language representations.

## Environment

```bash
conda activate severin
```

This is a SLURM cluster with H100 GPUs; you're likely on a login node — ask for instructions when GPU access is needed.

### Data

`data/flores_5lang_N1000.json` — 1000 parallel sentences across 5 languages (French, English, German, Dutch, Italian). Each entry has the same content in all languages, enabling clean comparison of SAE activations across languages to find language-specific features. **Use 0-800 for feature-finding, 800-1000 for eval.**

### Eval Datasets (all use FLORES 800-1000 or Dolly)
- `data/generation/` — "Answer in {lang}: {dolly_prompt}" — tests language production
- `data/generation_strict/` — "Answer only in {lang}: {dolly_prompt}" — stricter production test
- `data/do_not/` — "Do not answer in {lang}: {dolly_prompt}" — tests steering against instruction
- `data/comprehension/` — "Translate this to English: {text_in_lang}" — tests understanding of source language
- `data/translation/` — "Translate this to {lang}: {english_text}" — tests production via translation
- `data/translate_to/` — "{source}_to_{target}_200.json" — all 20 language pair permutations

### Keep Codebase Trim

Delete one-off scripts after use. If a script only generates an image or does a single analysis, remove it once complete. Keep only core, reusable code in the repository.

## Code Style

Python, PyTorch, SAELens. Minimalist, understandable code following clean research software design.

## Key Resources

### Notebooks (start here)
- `notebooks/steering_with_language_features.ipynb` — Main experiment notebook. Finds French-specific features via differential activation on FLORES+ parallel sentences, then demonstrates selective suppression of French production while preserving comprehension.
- `notebooks/SAETutorial_Gemma_Scope_2.ipynb` — Google's official Gemma Scope 2 tutorial (JumpReLU SAEs, hooks, steering)
- `notebooks/saelens_experiment_reference.md` — Comprehensive SAELens + Gemma Scope 2 API reference

### References
The `references/` folder contains:
- **Paper summaries** (markdown): "Unveiling Language-Specific Features..." and "Causal Language Control..."
- **multilingual-llm-features/** — Codebase from Deng et al. (ACL 2025). Useful for their monolinguality metric and ablation hooks on Gemma 2.

## Related Work Context

**Direct predecessors (extend their work):**
- Deng et al. "Unveiling Language-Specific Features in LLMs via SAEs" (ACL 2025) — Found 1-2 features per language are monolingual; ablating them degrades CE loss. Our contribution: separately measuring production vs comprehension.
- Chou et al. "Causal Language Control via Sparse Feature Steering" (ACL 2025 SRW) — Single-feature additive steering shifts output language. Complementary: they amplify, we ablate.

**Theoretical foundation:**
- Dumas et al. "Separating Tongue from Thought" (ACL 2025) — Early layers encode output language, later layers encode concepts. Explains why ablating language features in layer 22 kills production without touching comprehension.

**Methodological cautions:**
- Tang et al. "Language-Specific Neurons" (ACL 2024) — Anecdotally observed production/comprehension dissociation but never tested systematically.
- Le & Li "CRANE" (Jan 2026) — LAPE-identified neurons have near-zero causal relevance; highest-activating ≠ causally important. Consider whether our differential-activation selection might miss causally important features.
- Farrell et al. "SAEs for Unlearning" (NeurIPS ATTRIB 2024) — Zero ablation doesn't work; need negative scaling. Consistent with our projection suppression approach.
- Goncharov et al. "Language Steering in Latent Space" (2025) — Language identity is linearly separable and orthogonal to semantics; supports feasibility of surgical language ablation.

## Research Framing

This work fits into the **representation entanglement hierarchy**: behavioral dispositions are steerable, facts are partially entangled, capabilities are highly entangled. Language production/comprehension may sit at an intermediate level — production routes through specific SAE features, but comprehension is distributed across semantic representations.
