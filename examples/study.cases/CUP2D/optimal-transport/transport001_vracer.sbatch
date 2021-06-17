#!/bin/bash -l
#SBATCH --job-name="vracer_transport001"
#SBATCH --output=output_vracer_transport001_%j.out
#SBATCH --time=06:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=s929

cd ..
./batch-transport-cmaes.sh
