#!/bin/bash
#SBATCH --job-name="severin_sizes"
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=./logs/steering_eval.%j.out
#SBATCH --error=./logs/steering_eval.%j.err

# Note: 27B needs ~54GB VRAM (fits on 1 H100 80GB)
# If OOM, try: --gres=gpu:2 or a high-memory partition

module load miniforge3/24.3.0-0-gcc-11.5.0-wkw4vym

conda activate severin

cd /home/snfiel01/projects/saes_for_language

# Model configs: model_name, sae_release, layer (65% depth), d_model
# 1B:  26 layers, 65% = layer 17
# 4B:  34 layers, 65% = layer 22
# 27B: 62 layers, 65% = layer 40

declare -a CONFIGS=(
    "google/gemma-3-1b-it gemma-scope-2-1b-pt-res 17"
    "google/gemma-3-4b-it gemma-scope-2-4b-pt-res 22"
    "google/gemma-3-27b-it gemma-scope-2-27b-pt-res 40"
)

for config in "${CONFIGS[@]}"; do
    read -r model_name sae_release layer <<< "$config"

    echo "=== Model: ${model_name}, Layer: ${layer} ==="

    BASE_CMD="python scripts/run_intervention_on_eval.py \
        --intervention clamp \
        --language French \
        --strengths "0.0,0.1,0.2,0.5,1,2,5,10" \
        --n_samples 50 \
        --model_name ${model_name} \
        --sae_release ${sae_release} \
        --sae_id layer_${layer}_width_65k_l0_medium \
        --layer ${layer} \
        --keep_k 1 \
        --save_dir 4_clamp_various_model_sizes"

    echo "running comprehension"
    ${BASE_CMD} --eval_path comprehension/french_200.json --run_mmlu True
    echo "running generation"
    ${BASE_CMD} --eval_path generation/french_200.json
    echo "running generation_strict"
    ${BASE_CMD} --eval_path generation_strict/french_200.json
    echo "running translation"
    ${BASE_CMD} --eval_path translation/french_200.json
    echo "do NOT"
    ${BASE_CMD} --eval_path do_not/french_200.json
done
