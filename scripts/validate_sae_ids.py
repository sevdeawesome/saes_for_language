"""Validate SAE release names and IDs for our experiments.

Correct SAE naming from SAELens directory:
- Release: gemma-scope-2-{size}-pt-res  (NOT -resid_post)
- Widths: 16k, 65k, 262k, 1m  (NOT 64k)
- L0: small, medium, big

Available layers (only 4 per model, at ~25%, 50%, 65%, 85% depth):
- 1B (26 layers):  7, 13, 17, 22
- 4B (34 layers):  9, 17, 22, 29
- 27B (62 layers): 16, 31, 40, 53
"""

from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory


def print_available_saes():
    """Print all available SAE configurations for Gemma Scope 2."""
    saes_dir = get_pretrained_saes_directory()

    releases = [
        "gemma-scope-2-1b-pt-res",
        "gemma-scope-2-4b-pt-res",
        "gemma-scope-2-27b-pt-res",
    ]

    for release in releases:
        info = saes_dir[release]
        print(f"\n{'='*60}")
        print(f"Release: {release}")
        print(f"{'='*60}")

        # Group by layer
        layers = {}
        for sae_id in info.saes_map.keys():
            layer = int(sae_id.split("_")[1])
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(sae_id)

        print(f"Available layers: {sorted(layers.keys())}")
        print(f"\nSample SAE IDs for each layer:")
        for layer in sorted(layers.keys()):
            # Just show the 65k medium one as example
            medium = [s for s in layers[layer] if "65k_l0_medium" in s]
            if medium:
                print(f"  Layer {layer}: {medium[0]}")


def get_correct_configs():
    """Print the correct configurations for our experiments."""
    print("\n" + "="*60)
    print("CORRECT CONFIGURATIONS FOR EXPERIMENTS")
    print("="*60)

    print("\nExperiment 2: Various Layers (4B model)")
    print("-" * 40)
    print("Release: gemma-scope-2-4b-pt-res")
    print("Layers:  9, 17, 22, 29  (NOT 8!)")
    print("SAE ID format: layer_{N}_width_65k_l0_medium")

    print("\nExperiment 3: Various Model Sizes (~65% depth)")
    print("-" * 40)
    configs = [
        ("1B", "gemma-scope-2-1b-pt-res", 17, "~65% of 26 layers"),
        ("4B", "gemma-scope-2-4b-pt-res", 22, "~65% of 34 layers"),
        ("27B", "gemma-scope-2-27b-pt-res", 40, "~65% of 62 layers"),  # 40 is closer to 65% than 31
    ]
    for size, release, layer, note in configs:
        print(f"  {size}: {release}, layer_{layer}_width_65k_l0_medium ({note})")


if __name__ == "__main__":
    print("SAE Availability Check for Gemma Scope 2")
    print_available_saes()
    get_correct_configs()
    print("\n" + "="*60)
    print("Done!")
