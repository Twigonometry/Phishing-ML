#!/bin/bash
#SBATCH --comment=save_full_dataset
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --job-name=save_full_dataset
#SBATCH --output=save_full_dataset.log
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
# export WANDB_PROJECT=save_full_dataset
export TRANSFORMERS_CACHE=/mnt/parscratch/users/$USERNAME/cache

python save_full_dataset.py --save_path=/mnt/parscratch/users/hardcoded/full_dataset_dict --data_path=/mnt/parscratch/users/hardcoded/dfAll_for_liam/data.pkl