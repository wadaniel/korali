#!/bin/bash -l
#SBATCH --job-name="fish64"
#SBATCH --output=output_fish64_%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=65
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=s929
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu

cd ..
./batch-fish-vracer.sh
