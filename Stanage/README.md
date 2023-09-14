# Stanage Scripts

Scripts for the [Sheffield Stanage HPC](https://docs.hpc.shef.ac.uk/en/latest/stanage/index.html)

## Setup for A100

Qlora/bitsandbytes etc does not work on H100 nodes, here is how it was successfully setup on A100:

```bash
[user@login2 [stanage] ~]$ srun --partition=gpu --qos=gpu --gres=gpu:a100:1 --mem=10G --pty bash
[user@gpu09 [stanage] ~]$ cd ~
[user@gpu09 [stanage] ~]$ rm -r -f .conda
[user@gpu09 [stanage] ~]$ module load Anaconda3/2022.10
[user@gpu09 [stanage] ~]$ module load cuDNN/8.8.0.121-CUDA-12.0.0
[user@gpu09 [stanage] ~]$ conda create -n a100 python=3.10
[user@gpu09 [stanage] ~]$ conda activate a100
(a100) python -m pip install torch torchvision
(a100) python -m pip install transformers scikit-learn evaluate wandb peft accelerate bitsandbytes
```

Also need to run

```bash
git clone https://github.com/metalcorebear/NRCLex
cd NRCLex
python3 setup.py install --user
```