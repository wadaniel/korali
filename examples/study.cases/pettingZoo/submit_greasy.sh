#!/bin/bash -l
#SBATCH --job-name=greasy_submit
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-core=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=s929
#SBATCH --gres=gpu:0,craynetwork:4

module load GREASY

export CRAY_CUDA_MPS=1
export CUDA_VISIBLE_DEVICES=0
export GPU_DEVICE_ORDINAL=0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export GREASY_NWORKERS_PER_NODE=$SLURM_NTASKS_PER_NODE

greasy submit_batch.sh
