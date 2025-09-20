#!/bin/bash
#SBATCH --mem=80GB
#SBATCH -o cameraReady_quevedo_cis_entropy_surprisal_cws_mean_max.out
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --account=dcs-acad6
#SBATCH --reservation=dcs-acad6
#SBATCH --qos=gpu

module load Anaconda3/2022.10
module load CUDA/10.1.243
source activate /mnt/parscratch/users/acq22zm/anaconda/.envs/surprisal
export HF_HOME=/mnt/parscratch/users/acq22zm/.cache

LD_LIBRARY_PATH=""

model="quevedo"

python ../mlp_implementations.py \
    --features cis_sentence_mean cis_sentence_max vanilla_entropy_sentence_mean vanilla_entropy_sentence_max surprisal_sentence_mean surprisal_sentence_max cws_sentence_mean cws_sentence_max\
    --model $model \
    --output_dir ../../sentence-level/cameraReady_quevedo\
    --remarks cameraReady_quevedo_cis_entropy_surprisal_cws_mean_max \
    --save
