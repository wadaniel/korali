#!/bin/bash -l
#SBATCH --job-name="windmillnodes"
#SBATCH --output=output_windmillnodes%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=65
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=s929
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu

cd ..
./batch-windmill-vracer.sh
