#!/bin/bash
#SBATCH --mem=80GB
#SBATCH -o cameraReady_lr_baseline_together.out
#SBATCH --time=01:00:00

module load Anaconda3/2022.10
module load CUDA/10.1.243
source activate /mnt/parscratch/users/acq22zm/anaconda/.envs/surprisal
export HF_HOME=/mnt/parscratch/users/acq22zm/.cache

LD_LIBRARY_PATH=""

model="lr"

python ../classifier.py \
    --features log_prob confidence oddballness \
    --model $model \
    --output_dir ../../cameraReady_lr_results \
    --remarks cameraReady_lr_baseline_all3 \
    --baseline "y"\
    --save
