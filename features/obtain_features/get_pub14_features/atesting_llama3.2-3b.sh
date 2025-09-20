#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=100GB
#SBATCH --qos=gpu
#SBATCH --account=dcs-acad6
#SBATCH --reservation=dcs-acad6
#SBATCH --time=1-00:00:00
#SBATCH -o metonymy_allFeatures_llama(3.2-3B).out
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
    "pub_14_metonymy" 
)


for task in "${tasks[@]}"; do

    echo "***********${task}***********"

    echo "----------------------Starting llama 3.2-3B-Instruct---------------------------"
    python rough_v3.py \
        --path_to_dice "../../../dataset/${task}/pub14_extracted_fullstop.csv" \
        --hf_model "meta-llama/Llama-3.2-3B-Instruct" \
        --cws_gamma 0.5 \
        --task "${task}"
    echo "Finished llama-3.2-3B-Instruct"
    echo "---"

done