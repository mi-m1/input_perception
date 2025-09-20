#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=90GB
#SBATCH --qos=gpu
#SBATCH --account=dcs-acad6
#SBATCH --reservation=dcs-acad6
#SBATCH --time=1-00:00:00
#SBATCH -o allFeatures_qwen2514binst1m.out
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

python atesting.py \
--path_to_dice "../../dataset/dice/dice_v3_with_IDs.csv" \
--hf_model "Qwen/Qwen2.5-14B-Instruct-1M" \
--cws_gamma 0.5 \

# echo "------------------- NEXT MODEL -------------------"
# echo ""

# python atesting.py \
# --path_to_dice "../../dataset/dice/dice_v3_with_IDs.csv" \
# --hf_model "meta-llama/Llama-3.1-8B-Instruct" \
# --cws_gamma 0.5 \