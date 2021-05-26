#!/bin/bash -l
#SBATCH --job-name="windmill5_100"
#SBATCH --output=output_windmill5_100%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=33
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=s929
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu

cd ..
./batch-windmill-vracer.sh
