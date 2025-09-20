#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu,gpu-h100
#SBATCH --mem=100GB
#SBATCH --qos=gpu
#SBATCH --time=1-00:00:00
#SBATCH -o conmec_allFeatures_llama(3.1-8B-inst).out
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
    "conmec"
)


log_base_dir="/mnt/parscratch/users/acq22zm/surprisal/scripts/lighteval_dice"

for task in "${tasks[@]}"; do

    echo "***********${task}***********"

    echo "----------------------Starting llama 3.1-8B-Instruct---------------------------"
    python atesting_conmec.py \
        --path_to_data "../../../dataset/${task}/${task}_balanced.csv" \
        --hf_model "meta-llama/Llama-3.1-8B-Instruct" \
        --cws_gamma 0.5 \
        --task "${task}"
    echo "Finished llama 3.1-8B-Instruct"
    echo "---"


    # echo "----------------------Starting llama 3.2-3B-Instruct---------------------------"
    # python atesting_conmec.py \
    #     --path_to_data "../../../dataset/${task}/${task}_balanced.csv" \
    #     --hf_model "meta-llama/Llama-3.2-3B-Instruct" \
    #     --cws_gamma 0.5 \
    #     --task "${task}"
    # echo "Finished llama-3.2-3B-Instruct"
    # echo "---"

    # echo "----------------------Starting llama 3.2-1B-Instruct---------------------------"
    # python atesting_conmec.py \
    #     --path_to_data "../../../dataset/${task}/${task}_balanced.csv" \
    #     --hf_model "meta-llama/Llama-3.2-1B-Instruct" \
    #     --cws_gamma 0.5 \
    #     --task "${task}"
    # echo "Finished llama-3.2-1B-Instruct"
done