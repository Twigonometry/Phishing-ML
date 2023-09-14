#!/bin/bash
#SBATCH --comment=baseline_model_full
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --job-name=baseline_model_full
#SBATCH --output=baseline_model_full.log
#SBATCH --time 1-24:00:00
#SBATCH --ntasks=1
#SBATCH --mail-user=email
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
export WANDB_PROJECT=baseline_model_full
export TRANSFORMERS_CACHE=/mnt/parscratch/users/hardcoded/cache

python baseline_model.py --data_path=/mnt/parscratch/users/hardcoded/full_dataset_dict --save_path=/mnt/parscratch/users/hardcoded/