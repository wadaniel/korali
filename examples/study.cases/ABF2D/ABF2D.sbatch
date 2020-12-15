#!/bin/bash -l
#SBATCH --job-name="ABF2D"
#SBATCH --output=/scratch/snx3000/smartin/ABF2D/log%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=s929
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu

#pip3 uninstall korali -y; pushd ~/korali; ./install; popd
cat run-gfpt.py
python3 ./run-gfpt.py --dir ${SCRATCH}/ABF2D/${SLURM_JOB_ID}
