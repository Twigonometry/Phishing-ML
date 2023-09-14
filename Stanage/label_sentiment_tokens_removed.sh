#!/bin/bash
#SBATCH --comment=label_sentiment_tokens_removed
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --job-name=label_sentiment_tokens_removed
#SBATCH --output=label_sentiment_tokens_removed.log
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
# export WANDB_PROJECT=label_sentiment_tokens_removed
# export TRANSFORMERS_CACHE=/mnt/parscratch/users/$USERNAME/cache

python label_sentiment.py --train_data_path=/users/hardcoded/Stanage/mydatabase.emails.csv --test_data_path=/mnt/parscratch/users/hardcoded/dfAll_for_liam/data.pkl --save_path=/mnt/parscratch/users/hardcoded/sentiment_labelled_df_removed --replace_tokens