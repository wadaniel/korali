#!/bin/bash -l
#SBATCH --job-name="fish0032"
#SBATCH --output=fish0032_%j.out
#SBATCH --time=1-00:00:00
#SBATCH --nodes=33
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=s929
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu

cd ..
cat run-korali.cpp
./run_test.sh
