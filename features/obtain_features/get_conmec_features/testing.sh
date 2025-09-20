#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --partition=gpu,gpu-h100
#SBATCH --qos=gpu
#SBATCH --time=01:00:00
#SBATCH -o extraction_inspection.out

module load Anaconda3/2022.10
module load CUDA/10.1.243
module load GCC/12.3.0
source activate /mnt/parscratch/users/acq22zm/anaconda/.envs/surprisal
export HF_HOME=/mnt/parscratch/users/acq22zm/.cache
export HF_TOKEN=your_access_token

LD_LIBRARY_PATH=""

python atesting_conmec.py \

    