#!/bin/bash
#SBATCH --mem=80GB
#SBATCH -o cameraReady_ablating_surprisal.out
#SBATCH --time=01:00:00


module load Anaconda3/2022.10
module load CUDA/10.1.243
source activate /mnt/parscratch/users/acq22zm/anaconda/.envs/surprisal
export HF_HOME=/mnt/parscratch/users/acq22zm/.cache

LD_LIBRARY_PATH=""

model="lr"

python ../classifier.py \
    --features cis_sentence_mean cis_sentence_max vanilla_entropy_sentence_mean vanilla_entropy_sentence_max cws_sentence_mean cws_sentence_max\
    --model $model \
    --output_dir ../../cameraReady_lr_results \
    --remarks cameraReady_ablating_surprisal \
    --save
