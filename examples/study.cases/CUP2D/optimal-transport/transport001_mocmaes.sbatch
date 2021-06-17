#!/bin/bash -l
#SBATCH --job-name="mocmaes_transport001"
#SBATCH --output=output_mocmaes_transport001_%j.out
#SBATCH --time=12:00:00
#SBATCH --nodes=33
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=s929

cd ..
./batch-transport-mocmaes.sh
