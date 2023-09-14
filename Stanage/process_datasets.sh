#!/bin/bash
#SBATCH --comment=process_datasets
#SBATCH --partition=gpu-h100
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --job-name=process_datasets
#SBATCH --output=process_datasets.log
#SBATCH --time 0-24:00:00
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
# export WANDB_PROJECT=process_datasets
export TRANSFORMERS_CACHE=/mnt/parscratch/users/$USERNAME/cache

python process_datasets.py