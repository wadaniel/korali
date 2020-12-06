#!/bin/bash -l
#SBATCH --job-name="fish0004"
#SBATCH --output=fish0004_%j.out
#SBATCH --time=2:00:00
#SBATCH --nodes=5
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=s929
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu

cd ..
cat run-korali.cpp
./run_test.sh
