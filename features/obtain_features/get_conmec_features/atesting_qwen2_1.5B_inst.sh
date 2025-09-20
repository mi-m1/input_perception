#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu,gpu-h100
#SBATCH --mem=100GB
#SBATCH --qos=gpu
#SBATCH --time=1-00:00:00
#SBATCH -o conmec_qwen(2-1.5B-Inst).out
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
    "conmec"
)


# Loop through the array and print each name
for task in "${tasks[@]}"; do

    echo "***********${task}***********"
    
    # echo "----------------------Starting Qwen/Qwen2.5-0.5B-Instruct---------------------------"
    # python atesting_conmec.py \
    # --path_to_data "../../../dataset/${task}/${task}_balanced.csv" \
    # --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
    # --cws_gamma 0.5 \
    # --task "${task}"
    # echo "Finished Qwen/Qwen2.5-0.5B-Instruct"

    echo "----------------------Starting Qwen/Qwen2-1.5B-Instruct---------------------------"
    python atesting_conmec.py \
    --path_to_data "../../../dataset/${task}/${task}_balanced.csv" \
    --hf_model "Qwen/Qwen2-1.5B-Instruct" \
    --cws_gamma 0.5 \
    --task "${task}"
    echo "Finished Qwen/Qwen2-1.5B-Instruct"

    # echo "----------------------Starting Qwen/Qwen2.5-7B-Instruct-1M---------------------------"
    # python atesting_conmec.py \
    # --path_to_data "../../../dataset/${task}/${task}_balanced.csv" \
    # --hf_model "Qwen/Qwen2.5-7B-Instruct-1M" \
    # --cws_gamma 0.5 \
    # --task "${task}"
    # echo "Finished Qwen/Qwen2.5-7B-Instruct-1M"

    # echo "----------------------Starting Qwen/Qwen2.5-14B-Instruct-1M---------------------------"
    # python atesting_conmec.py \
    # --path_to_data "../../../dataset/${task}/${task}_balanced.csv" \
    # --hf_model "Qwen/Qwen2.5-14B-Instruct-1M" \
    # --cws_gamma 0.5 \
    # --task "${task}"
    # echo "Finished Qwen/Qwen2.5-14B-Instruct-1M"

done



