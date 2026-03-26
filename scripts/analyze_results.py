#!/usr/bin/env python3
"""Analyze experiment results and generate figures for the report.

Produces:
- Language detection metrics (% French in generation outputs)
- Comprehension preservation metrics
- Figures: production suppression, comprehension preservation, layer comparison, model size comparison, clamping
- Curated interesting/funny generations
"""

import json
import glob
import re
import os
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

try:
    from lingua import Language, LanguageDetectorBuilder
    detector = LanguageDetectorBuilder.from_languages(
        Language.FRENCH, Language.ENGLISH, Language.GERMAN,
        Language.DUTCH, Language.ITALIAN, Language.THAI,
        Language.SPANISH, Language.PORTUGUESE
    ).build()
    USE_LINGUA = True
except Exception:
    USE_LINGUA = False

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# ============================================================
# UTILS
# ============================================================

def detect_language(text):
    """Return detected language string or 'unknown'."""
    if not text or len(text.strip()) < 10:
        return "unknown"
    # Strip common preambles like "Okay, here's..."
    # Try to get to the actual content
    text_clean = text.strip()

    # Check for degenerate repetitions
    words = text_clean.split()
    if len(words) > 5:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.1:
            return "degenerate"

    if USE_LINGUA:
        result = detector.detect_language_of(text_clean)
        if result:
            return result.name.lower()

    return "unknown"


def french_fraction(generations):
    """Fraction of generations detected as French."""
    if not generations:
        return 0.0
    n_french = 0
    for g in generations:
        lang = detect_language(g["generation"])
        if lang == "french":
            n_french += 1
    return n_french / len(generations)


def english_fraction(generations):
    """Fraction of generations detected as English."""
    if not generations:
        return 0.0
    n_eng = 0
    for g in generations:
        lang = detect_language(g["generation"])
        if lang == "english":
            n_eng += 1
    return n_eng / len(generations)


def degenerate_fraction(generations):
    """Fraction of generations that are degenerate (repetitive loops)."""
    if not generations:
        return 0.0
    n_deg = 0
    for g in generations:
        lang = detect_language(g["generation"])
        if lang == "degenerate":
            n_deg += 1
    return n_deg / len(generations)


def comprehension_score(generations):
    """Simple proxy: fraction of comprehension outputs detected as English.
    For 'Translate this to English' prompts, a good model outputs English."""
    return english_fraction(generations)


def load_result(path):
    with open(path) as f:
        return json.load(f)


# ============================================================
# EXPERIMENT 1: ABLATION - VARIOUS TOP-K
# ============================================================

def analyze_topk():
    """Analyze ablation results across different top-k feature counts."""
    print("\n=== Experiment 1: Ablation with various top-k ===")

    tasks = ["generation", "generation_strict", "comprehension", "translation"]
    topk_values = [1, 2, 3, 5, 10, 15, 30]

    # Collect data per task per topk
    # Structure: task -> topk -> [(strength, french_frac, english_frac, degen_frac)]
    data = defaultdict(lambda: defaultdict(list))

    for task in tasks:
        for k in topk_values:
            path = RESULTS_DIR / f"1_various_topk/ablate_French_gemma3_4b_layer29_{task}_{k}.json"
            if not path.exists():
                continue
            result = load_result(path)
            for r in result["results"]:
                s = r["strength"]
                gens = r["generations"]
                data[task][k].append({
                    "strength": s,
                    "french": french_fraction(gens),
                    "english": english_fraction(gens),
                    "degenerate": degenerate_fraction(gens),
                })

    return data


def plot_topk_production_vs_comprehension(data):
    """Main figure: production suppression vs comprehension preservation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 7))
    topk_values = [1, 2, 3, 5, 10, 15, 30]

    # Left: Generation (production) - French fraction should drop
    ax = axes[0]
    for i, k in enumerate(topk_values):
        if k not in data["generation"]:
            continue
        entries = sorted(data["generation"][k], key=lambda x: x["strength"])
        strengths = [e["strength"] for e in entries]
        fr_fracs = [e["french"] * 100 for e in entries]
        ax.plot(strengths, fr_fracs, "o-", color=colors[i], label=f"k={k}", linewidth=2, markersize=6)

    ax.set_xlabel("Ablation Strength", fontsize=12)
    ax.set_ylabel("% French Output", fontsize=12)
    ax.set_title('Production: "Answer in French: ..."', fontsize=13, fontweight="bold")
    ax.legend(title="# features ablated", fontsize=9)
    ax.set_ylim(-5, 105)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.grid(True, alpha=0.3)

    # Right: Comprehension - English fraction should stay high
    ax = axes[1]
    for i, k in enumerate(topk_values):
        if k not in data["comprehension"]:
            continue
        entries = sorted(data["comprehension"][k], key=lambda x: x["strength"])
        strengths = [e["strength"] for e in entries]
        en_fracs = [e["english"] * 100 for e in entries]
        ax.plot(strengths, en_fracs, "o-", color=colors[i], label=f"k={k}", linewidth=2, markersize=6)

    ax.set_xlabel("Ablation Strength", fontsize=12)
    ax.set_ylabel("% English Output", fontsize=12)
    ax.set_title('Comprehension: "Translate French → English"', fontsize=13, fontweight="bold")
    ax.legend(title="# features ablated", fontsize=9)
    ax.set_ylim(-5, 105)
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.3)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Ablating French SAE Features: Production vs Comprehension\n(Gemma 3 4B, Layer 29)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_production_vs_comprehension.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fig1_production_vs_comprehension.png")
    plt.close()


def plot_topk_all_tasks(data):
    """4-panel figure showing all tasks across strengths for k=1."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    tasks = ["generation", "generation_strict", "comprehension", "translation"]
    titles = [
        '"Answer in French: ..."',
        '"Answer ONLY in French: ..."',
        '"Translate French → English"',
        '"Translate English → French"',
    ]
    ylabels = ["% French", "% French", "% English", "% French"]
    metrics = ["french", "french", "english", "french"]

    k = 1  # Focus on single feature

    for idx, (task, title, ylabel, metric) in enumerate(zip(tasks, titles, ylabels, metrics)):
        ax = axes[idx // 2][idx % 2]
        if k not in data[task]:
            continue
        entries = sorted(data[task][k], key=lambda x: x["strength"])
        strengths = [e["strength"] for e in entries]
        vals = [e[metric] * 100 for e in entries]
        degen = [e["degenerate"] * 100 for e in entries]

        ax.plot(strengths, vals, "o-", color="#2196F3", linewidth=2.5, markersize=7, label=ylabel)
        ax.plot(strengths, degen, "s--", color="#F44336", linewidth=1.5, markersize=5, label="% Degenerate")

        ax.set_xlabel("Ablation Strength", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(-5, 105)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Single French Feature Ablation Across All Tasks\n(Gemma 3 4B, Layer 29, k=1, Feature #205)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_all_tasks_k1.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fig2_all_tasks_k1.png")
    plt.close()


# ============================================================
# EXPERIMENT 2: VARIOUS LAYERS
# ============================================================

def analyze_and_plot_layers():
    """Analyze ablation at different layers."""
    print("\n=== Experiment 2: Various Layers ===")

    layers = [9, 17, 22, 29]
    tasks = ["generation", "generation_strict", "comprehension", "translation"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = {"generation": "#2196F3", "generation_strict": "#4CAF50",
              "comprehension": "#FF9800", "translation": "#9C27B0"}

    # For each layer, compute French% at strength=0 and max strength for generation
    layer_gen_data = {}
    layer_comp_data = {}

    for layer in layers:
        for task in tasks:
            path = RESULTS_DIR / f"2_various_layers/ablate_French_gemma3_4b_layer{layer}_{task}_1.json"
            if not path.exists():
                continue
            result = load_result(path)
            for r in result["results"]:
                s = r["strength"]
                gens = r["generations"]
                if task in ["generation", "generation_strict"]:
                    if layer not in layer_gen_data:
                        layer_gen_data[layer] = {}
                    if task not in layer_gen_data[layer]:
                        layer_gen_data[layer][task] = []
                    layer_gen_data[layer][task].append((s, french_fraction(gens) * 100))
                elif task == "comprehension":
                    if layer not in layer_comp_data:
                        layer_comp_data[layer] = []
                    layer_comp_data[layer].append((s, english_fraction(gens) * 100))

    # Left: Production across layers
    ax = axes[0]
    layer_colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(layers)))
    for i, layer in enumerate(layers):
        if layer in layer_gen_data and "generation" in layer_gen_data[layer]:
            pts = sorted(layer_gen_data[layer]["generation"])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                    color=layer_colors[i], label=f"Layer {layer}", linewidth=2, markersize=6)
    ax.set_xlabel("Ablation Strength", fontsize=12)
    ax.set_ylabel("% French Output", fontsize=12)
    ax.set_title("Production Suppression by Layer", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    # Right: Comprehension across layers
    ax = axes[1]
    for i, layer in enumerate(layers):
        if layer in layer_comp_data:
            pts = sorted(layer_comp_data[layer])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                    color=layer_colors[i], label=f"Layer {layer}", linewidth=2, markersize=6)
    ax.set_xlabel("Ablation Strength", fontsize=12)
    ax.set_ylabel("% English Output", fontsize=12)
    ax.set_title("Comprehension Preservation by Layer", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Layer Comparison: Where Do Language Features Live?\n(Gemma 3 4B, k=1)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_layer_comparison.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fig3_layer_comparison.png")
    plt.close()


# ============================================================
# EXPERIMENT 3: VARIOUS MODEL SIZES
# ============================================================

def analyze_and_plot_model_sizes():
    """Analyze ablation across 1B, 4B, 27B models."""
    print("\n=== Experiment 3: Various Model Sizes ===")

    models = [
        ("gemma3_1b", "layer17", "Gemma 3 1B"),
        ("gemma3_4b", "layer22", "Gemma 3 4B"),
        ("gemma3_27b", "layer40", "Gemma 3 27B"),
    ]
    tasks = ["generation", "generation_strict", "comprehension", "translation"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    model_colors = ["#E91E63", "#2196F3", "#4CAF50"]

    for task_idx, task in enumerate(tasks):
        ax = axes[task_idx // 2][task_idx % 2]
        metric = "english" if task == "comprehension" else "french"
        ylabel = "% English" if task == "comprehension" else "% French"

        for m_idx, (model_code, layer_str, model_label) in enumerate(models):
            path = RESULTS_DIR / f"3_various_model_sizes/ablate_French_{model_code}_{layer_str}_{task}_5.json"
            if not path.exists():
                continue
            result = load_result(path)
            pts = []
            for r in result["results"]:
                s = r["strength"]
                gens = r["generations"]
                if metric == "french":
                    val = french_fraction(gens) * 100
                else:
                    val = english_fraction(gens) * 100
                pts.append((s, val))
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                    color=model_colors[m_idx], label=model_label, linewidth=2, markersize=6)

        titles = {
            "generation": '"Answer in French"',
            "generation_strict": '"Answer ONLY in French"',
            "comprehension": '"Translate French → English"',
            "translation": '"Translate English → French"',
        }
        ax.set_xlabel("Ablation Strength", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(titles[task], fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Model Size Comparison: Ablation of Top-5 French Features\n(Layer ~85% depth for each model)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_model_size_comparison.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fig4_model_size_comparison.png")
    plt.close()


# ============================================================
# EXPERIMENT 4: CLAMPING (STEERING)
# ============================================================

def analyze_and_plot_clamping():
    """Analyze clamping/steering results."""
    print("\n=== Experiment 4: Clamping (Steering) ===")

    models = [
        ("gemma3_1b", "layer17", "Gemma 3 1B"),
        ("gemma3_4b", "layer22", "Gemma 3 4B"),
        ("gemma3_27b", "layer40", "Gemma 3 27B"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    model_colors = ["#E91E63", "#2196F3", "#4CAF50"]

    # Panel 1: French % on "do_not" task (should increase with clamping)
    ax = axes[0]
    for m_idx, (model_code, layer_str, model_label) in enumerate(models):
        path = RESULTS_DIR / f"4_clamp_various_model_sizes/clamp_French_{model_code}_{layer_str}_do_not_1.json"
        if not path.exists():
            continue
        result = load_result(path)
        pts = []
        for r in result["results"]:
            s = r["strength"]
            gens = r["generations"]
            fr = french_fraction(gens) * 100
            dg = degenerate_fraction(gens) * 100
            pts.append((s, fr, dg))
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                color=model_colors[m_idx], label=model_label, linewidth=2, markersize=6)
    ax.set_xlabel("Clamp Strength", fontsize=12)
    ax.set_ylabel("% French Output", fontsize=12)
    ax.set_title('"Do NOT answer in French"', fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    # Panel 2: Generation (already asked in French) + clamping
    ax = axes[1]
    for m_idx, (model_code, layer_str, model_label) in enumerate(models):
        path = RESULTS_DIR / f"4_clamp_various_model_sizes/clamp_French_{model_code}_{layer_str}_generation_1.json"
        if not path.exists():
            continue
        result = load_result(path)
        pts = []
        for r in result["results"]:
            s = r["strength"]
            gens = r["generations"]
            dg = degenerate_fraction(gens) * 100
            pts.append((s, dg))
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "s--",
                color=model_colors[m_idx], label=model_label, linewidth=2, markersize=6)
    ax.set_xlabel("Clamp Strength", fontsize=12)
    ax.set_ylabel("% Degenerate Output", fontsize=12)
    ax.set_title("Degeneration at High Clamp Strength", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    # Panel 3: Comprehension with clamping
    ax = axes[2]
    for m_idx, (model_code, layer_str, model_label) in enumerate(models):
        path = RESULTS_DIR / f"4_clamp_various_model_sizes/clamp_French_{model_code}_{layer_str}_comprehension_1.json"
        if not path.exists():
            continue
        result = load_result(path)
        pts = []
        for r in result["results"]:
            s = r["strength"]
            gens = r["generations"]
            en = english_fraction(gens) * 100
            pts.append((s, en))
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                color=model_colors[m_idx], label=model_label, linewidth=2, markersize=6)
    ax.set_xlabel("Clamp Strength", fontsize=12)
    ax.set_ylabel("% English Output", fontsize=12)
    ax.set_title("Comprehension Under Clamping", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Clamping (Steering Toward French): Can We Override Instructions?\n(Top-1 French feature, various model sizes)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_clamping.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fig5_clamping.png")
    plt.close()


# ============================================================
# SUMMARY TABLE
# ============================================================

def generate_summary_table(topk_data):
    """Print a summary table for the report."""
    print("\n=== Summary Table (k=1, Gemma 3 4B, Layer 29) ===")
    print(f"{'Strength':>10} | {'Gen %Fr':>8} | {'GenStrict %Fr':>14} | {'Comp %En':>9} | {'Trans %Fr':>10}")
    print("-" * 65)

    tasks_data = {}
    for task in ["generation", "generation_strict", "comprehension", "translation"]:
        if 1 in topk_data[task]:
            entries = sorted(topk_data[task][1], key=lambda x: x["strength"])
            tasks_data[task] = {e["strength"]: e for e in entries}

    strengths = sorted(set(s for td in tasks_data.values() for s in td.keys()))

    rows = []
    for s in strengths:
        gen_fr = tasks_data.get("generation", {}).get(s, {}).get("french", 0) * 100
        gs_fr = tasks_data.get("generation_strict", {}).get(s, {}).get("french", 0) * 100
        comp_en = tasks_data.get("comprehension", {}).get(s, {}).get("english", 0) * 100
        trans_fr = tasks_data.get("translation", {}).get(s, {}).get("french", 0) * 100
        print(f"{s:>10.2f} | {gen_fr:>7.1f}% | {gs_fr:>13.1f}% | {comp_en:>8.1f}% | {trans_fr:>9.1f}%")
        rows.append((s, gen_fr, gs_fr, comp_en, trans_fr))

    return rows


# ============================================================
# CURATED EXAMPLES
# ============================================================

def find_interesting_generations():
    """Find funny/interesting generation examples."""
    print("\n=== Finding interesting generations ===")

    examples = []

    # 1. Production suppression at medium strength (English fallback)
    path = RESULTS_DIR / "1_various_topk/ablate_French_gemma3_4b_layer29_generation_1.json"
    if path.exists():
        result = load_result(path)
        for r in result["results"]:
            if r["strength"] == 1.0:
                for g in r["generations"]:
                    # Look for interesting cases where it switches mid-generation
                    gen = g["generation"]
                    if ("French" in gen[:50] and not any(c in gen[50:150] for c in "àéèêëîïôùûç")):
                        examples.append({
                            "category": "production_suppression_clean",
                            "strength": r["strength"],
                            "experiment": "ablate k=1 layer 29",
                            "prompt": g["prompt"],
                            "generation": gen[:300],
                        })
            if r["strength"] == 6.0:
                for g in r["generations"]:
                    gen = g["generation"]
                    # Look for Thai/Khmer gibberish or "inherently" repetition
                    if "inherently" in gen.lower() or any(ord(c) > 3000 for c in gen):
                        examples.append({
                            "category": "production_suppression_extreme",
                            "strength": r["strength"],
                            "experiment": "ablate k=1 layer 29",
                            "prompt": g["prompt"],
                            "generation": gen[:300],
                        })

    # 2. Translation suppression (can't produce French, outputs other scripts)
    path = RESULTS_DIR / "1_various_topk/ablate_French_gemma3_4b_layer29_translation_1.json"
    if path.exists():
        result = load_result(path)
        for r in result["results"]:
            if r["strength"] == 6.0:
                for g in r["generations"]:
                    gen = g["generation"]
                    if any(ord(c) > 3000 for c in gen) or "inherently" in gen.lower():
                        examples.append({
                            "category": "translation_broken",
                            "strength": r["strength"],
                            "experiment": "ablate k=1 layer 29",
                            "prompt": g["prompt"],
                            "generation": gen[:300],
                        })

    # 3. Clamping degenerate outputs
    path = RESULTS_DIR / "4_clamp_various_model_sizes/clamp_French_gemma3_4b_layer22_generation_1.json"
    if path.exists():
        result = load_result(path)
        for r in result["results"]:
            if r["strength"] == 0.2:
                for g in r["generations"][:5]:
                    examples.append({
                        "category": "clamp_garbled_french",
                        "strength": r["strength"],
                        "experiment": "clamp k=1 layer 22",
                        "prompt": g["prompt"],
                        "generation": g["generation"][:300],
                    })
                break
            if r["strength"] == 1.0:
                for g in r["generations"][:3]:
                    examples.append({
                        "category": "clamp_retour_loop",
                        "strength": r["strength"],
                        "experiment": "clamp k=1 layer 22",
                        "prompt": g["prompt"],
                        "generation": g["generation"][:200],
                    })
                break

    # 4. do_not with clamping (overriding instruction)
    path = RESULTS_DIR / "4_clamp_various_model_sizes/clamp_French_gemma3_4b_layer22_do_not_1.json"
    if path.exists():
        result = load_result(path)
        for r in result["results"]:
            if r["strength"] == 0.2:
                for g in r["generations"][:3]:
                    examples.append({
                        "category": "clamp_override_instruction",
                        "strength": r["strength"],
                        "experiment": "clamp k=1 layer 22 (do_not)",
                        "prompt": g["prompt"],
                        "generation": g["generation"][:300],
                    })
                break

    # 5. Comprehension preserved even at extreme ablation
    path = RESULTS_DIR / "1_various_topk/ablate_French_gemma3_4b_layer29_comprehension_1.json"
    if path.exists():
        result = load_result(path)
        for r in result["results"]:
            if r["strength"] == 6.0:
                for g in r["generations"][:3]:
                    examples.append({
                        "category": "comprehension_preserved",
                        "strength": r["strength"],
                        "experiment": "ablate k=1 layer 29",
                        "prompt": g["prompt"],
                        "generation": g["generation"][:300],
                    })
                break

    return examples


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)

    topk_data = analyze_topk()
    plot_topk_production_vs_comprehension(topk_data)
    plot_topk_all_tasks(topk_data)
    summary_rows = generate_summary_table(topk_data)

    analyze_and_plot_layers()
    analyze_and_plot_model_sizes()
    analyze_and_plot_clamping()

    examples = find_interesting_generations()

    # Save examples for report
    with open("figures/curated_examples.json", "w") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved {len(examples)} curated examples to figures/curated_examples.json")

    # Save summary table
    with open("figures/summary_table.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    print("\n=== Done! All figures saved to figures/ ===")
