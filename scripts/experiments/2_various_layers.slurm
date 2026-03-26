#!/bin/bash
#SBATCH --job-name="severin_layers"
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/steering_eval.%j.out
#SBATCH --error=./logs/steering_eval.%j.err


module load miniforge3/24.3.0-0-gcc-11.5.0-wkw4vym

conda activate severin

cd /home/snfiel01/projects/saes_for_language

for layer in 9 17 22 29; do
    BASE_CMD="python scripts/run_intervention_on_eval.py \
        --intervention ablate \
        --language French \
        --strengths \"0.0,0.25,0.5,1,2,6\" \
        --n_samples 50 \
        --model_name google/gemma-3-4b-it \
        --sae_release gemma-scope-2-4b-pt-res \
        --sae_id layer_${layer}_width_65k_l0_medium \
        --layer ${layer} \
        --keep_k 1 \
        --save_dir 2_various_layers"

    echo "=== Layer ${layer} ==="
    echo "running comprehension"
    ${BASE_CMD} --eval_path comprehension/french_200.json --run_mmlu True
    echo "running generation"
    ${BASE_CMD} --eval_path generation/french_200.json
    echo "running generation_strict"
    ${BASE_CMD} --eval_path generation_strict/french_200.json
    echo "running translation"
    ${BASE_CMD} --eval_path translation/french_200.json
    echo "running translation german --> french"
    ${BASE_CMD} --eval_path translate_to/german_to_french.json
    echo "running translation french --> german"
    ${BASE_CMD} --eval_path translate_to/french_to_german.json
done
