#!/bin/bash -l
#SBATCH --job-name="cmaes_transport001"
#SBATCH --output=output_cmaes_transport001_%j.out
#SBATCH --time=08:00:00
#SBATCH --nodes=33
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=s929

./batch-transport-cmaes.sh
