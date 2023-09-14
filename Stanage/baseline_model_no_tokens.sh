#!/bin/bash
#SBATCH --comment=baseline_model_no_tokens
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --job-name=baseline_model_no_tokens
#SBATCH --output=baseline_model_no_tokens.log
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
conda activate pytorch

# run your custom scripts:
export WANDB_PROJECT=baseline_model_no_tokens
export TRANSFORMERS_CACHE=/mnt/parscratch/users/$USERNAME/cache

python baseline_model.py --data_path=/mnt/parscratch/users/hardcoded/dis_dataset_dict_no_tokens --save_path=/mnt/parscratch/users/hardcoded/no_tokens_