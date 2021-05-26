#!/bin/bash -l
#SBATCH --job-name="windmill_test"
#SBATCH --output=output_windmill_test%j.out
#SBATCH --time=3:00:00
#SBATCH --nodes=3
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=s929
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu

cd ..

./batch-pair-windmill-vracer.sh
