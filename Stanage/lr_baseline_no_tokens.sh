#!/bin/bash
#SBATCH --comment=lr_baseline_no_tokens
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --job-name=lr_baseline_no_tokens
#SBATCH --output=lr_baseline_no_tokens.log
#SBATCH --time 1-24:00:00
#SBATCH --ntasks=1
#SBATCH --mail-user=$EMAIL
#SBATCH --mail-type=ALL

# load modules or conda environments here
module load Anaconda3/2022.10
module load cuDNN/8.8.0.121-CUDA-12.0.0

# Check CUDA loaded correctly and GPU status
nvcc --version
nvidia-smi

source /opt/apps/testapps/common/software/staging/Anaconda3/2022.10/bin/activate
conda activate a100

# run your custom scripts:
# export WANDB_PROJECT=lr_baseline
export TRANSFORMERS_CACHE=/mnt/parscratch/users/$USERNAME/cache

python lr_baseline.py --data_path=/mnt/parscratch/users/hardcoded/dis_dataset_dict_no_tokens --save_path=/mnt/parscratch/users/hardcoded/lr_baseline_no_tokens_accuracy