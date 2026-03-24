#!/bin/bash
#SBATCH --job-name="severin_j"
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/steering_eval.%j.out
#SBATCH --error=./logs/steering_eval.%j.err


module load miniforge3/24.3.0-0-gcc-11.5.0-wkw4vym

conda activate severin

cd /home/snfiel01/projects/saes_for_language



BASE_CMD="python scripts/run_intervention_on_eval.py \
    --intervention ablate \
    --language French \
    --strengths \"0.0,1,2,3,4,5,6\" \
    --n_samples 50 \
    --model_name google/gemma-3-4b-it \
    --sae_release gemma-scope-2-4b-pt-res \
    --sae_id layer_29_width_65k_l0_medium \
    --layer 29 \
    --save_dir 1_various_topk"

#${BASE_CMD} --eval_path comprehension/french_200.json --keep_k 5
# ${BASE_CMD} --eval_path generation/french_200.json --keep_k 5
# ${BASE_CMD} --eval_path translation/french_200.json --keep_k 5

for keep_k in 1 2 3 4 5 6 7 8 9 10 15 20 30; do
    echo "running comprehension"
    ${BASE_CMD} --eval_path comprehension/french_200.json --keep_k ${keep_k} --run_mmlu True
    echo "running generation"
    ${BASE_CMD} --eval_path generation/french_200.json --keep_k ${keep_k}
    echo "running generation_strict"
    ${BASE_CMD} --eval_path generation_strict/french_200.json --keep_k ${keep_k}
    echo "running translation"
    ${BASE_CMD} --eval_path translation/french_200.json --keep_k ${keep_k}
done