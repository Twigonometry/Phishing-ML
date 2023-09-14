#!/bin/bash
#SBATCH --comment=t5_qlora
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --job-name=t5_qlora
#SBATCH --output=t5_qlora.log
#SBATCH --time 2-24:00:00
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
export WANDB_PROJECT=t5_qlora
export TRANSFORMERS_CACHE=/mnt/parscratch/users/$USERNAME/cache

python t5_qlora.py --data_path=/mnt/parscratch/users/hardcoded/dataset_dict --save_path=/mnt/parscratch/users/hardcoded/ --load_pt_from_file=/mnt/parscratch/users/hardcoded/gen_processed_dataset_dict_qlora.pt