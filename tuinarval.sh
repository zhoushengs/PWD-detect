#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:1  # request a GPU
#SBATCH --mem=40G      
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=2
#SBATCH --time=4:03:00
#SBATCH --output=%j-fasternet_dw.out
#SBATCH --account=def-xianyi

module load StdEnv/2020 gcc/9.3.0 opencv/4.7.0 "cuda/11.7" python/3.8.10
source ~/envs/yoloEnv/bin/activate
pip install --no-index --upgrade pip
nvidia-smi

echo "Job ID: $SLURM_JOB_ID"
echo "starting training..."

python -u train_VisDrone.py
#python -u train_on_cc.py
#python -u train_rtdetr.py       #   -m torch.distributed.run --nproc_per_node 3 train_rtdetr.py 
#python -u -m torch.distributed.run --nproc_per_node 3 train_VisDrone.py