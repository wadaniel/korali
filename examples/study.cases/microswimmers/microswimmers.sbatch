#!/bin/bash -l
#SBATCH --job-name="microswimmers"
#SBATCH --output=/scratch/snx3000/smartin/microswimmers/log%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=s929
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu

#pip3 uninstall korali -y; pushd ~/korali; ./install; popd
cat run-korali.cpp
./run-korali --dir ${SCRATCH}/microswimmers/${SLURM_JOB_ID}
