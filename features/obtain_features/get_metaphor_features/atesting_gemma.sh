#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --account=dcs-acad6
#SBATCH --reservation=dcs-acad6
#SBATCH --mem=100GB
#SBATCH --qos=gpu
#SBATCH --time=1-00:00:00
#SBATCH -o metaphor_allFeatures_gemma.out
#SBATCH --mail-user=zmi1@sheffield.ac.uk
#SBATCH --mail-type BEGIN
#SBATCH --mail-type END
#SBATCH --mail-type FAIL

module load Anaconda3/2022.10
module load CUDA/10.1.243
source activate /mnt/parscratch/users/acq22zm/anaconda/.envs/surprisal
export HF_HOME=/mnt/parscratch/users/acq22zm/.cache
export HF_TOKEN=your_access_token

LD_LIBRARY_PATH=""


tasks=(
    # "custom|dice|0|0"
    "custom|mohx|0|0"
    "custom|trofi|0|0"
)

log_base_dir="/mnt/parscratch/users/acq22zm/surprisal/scripts/lighteval_dice"

for task in "${tasks[@]}"; do
    task_name=$(echo "$task" | cut -d'|' -f2)
    echo "***********${task_name}***********"

    python atesting_metaphor.py \
        --path_to_dice "../../../dataset/TroFi/TroFi_labelCorrected_withIDs.csv" \
        --hf_model "google/gemma-2b-it" \
        --cws_gamma 0.5 \
        --task "${task}"
    echo "Finished gemma-2b-it"
    echo "---"
    
    python atesting_metaphor.py \
        --path_to_dice "../../../dataset/TroFi/TroFi_labelCorrected_withIDs.csv" \
        --hf_model "google/gemma-7b-it" \
        --cws_gamma 0.5 \
        --task "${task}"
    echo "Finished gemma-7b-it"
    echo "---"
done