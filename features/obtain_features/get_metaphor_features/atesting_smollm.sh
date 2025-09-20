#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --account=dcs-acad6
#SBATCH --reservation=dcs-acad6
#SBATCH --mem=100GB
#SBATCH --qos=gpu
#SBATCH --time=1-00:00:00
#SBATCH -o metaphor_allFeatures_smollm.out
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


# Define an array of names
tasks=(
    "MOH-X" 
    "TroFi" 
)


# Loop through the array and print each name
for task in "${tasks[@]}"; do

    echo "***********${task}***********"
    echo "----------------------Starting SmolLM-1.7B-Instruct---------------------------"
    python atesting_metaphor.py \
    --path_to_dice "../../../dataset/${task}/${task}_labelCorrected_withIDs.csv" \
    --hf_model "HuggingFaceTB/SmolLM-1.7B-Instruct" \
    --cws_gamma 0.5 \
    --task "${task}"
    echo "Finished SmolLM-1.7B-Instruct"

    echo "----------------------Starting SmolLM-360M-Instruct---------------------------"
    python atesting_metaphor.py \
    --path_to_dice "../../../dataset/${task}/${task}_labelCorrected_withIDs.csv" \
    --hf_model "HuggingFaceTB/SmolLM-360M-Instruct" \
    --cws_gamma 0.5 \
    --task "${task}"
    echo "Finished "SmolLM-360M-Instruct""

done



