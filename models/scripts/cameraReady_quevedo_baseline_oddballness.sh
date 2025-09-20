#!/bin/bash
#SBATCH --mem=80GB
#SBATCH -o cameraReady_quevedo_oddballness.out
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
    --features oddballness\
    --model $model \
    --output_dir ../../sentence-level/cameraReady_quevedo\
    --remarks cameraReady_quevedo_oddballness \
    --baseline "y"\
    --save
