#!/bin/bash
#SBATCH --comment=train_adversarial
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --job-name=train_adversarial
#SBATCH --output=train_adversarial.log
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
# export WANDB_PROJECT=train_adversarial
export TRANSFORMERS_CACHE=/mnt/parscratch/users/hardcoded/cache

python train_adversarial.py --dis_model_path=/mnt/parscratch/users/harcoded/baseline_model.out --epochs=10 --emails_per_epoch=50 --save_path=/mnt/parscratch/users/harcoded/