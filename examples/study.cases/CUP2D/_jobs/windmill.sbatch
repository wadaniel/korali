#!/bin/bash -l
#SBATCH --job-name="windmill_flow_normal"
#SBATCH --output=output_windmill_energy_normal%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=65
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=s929
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu

cd ..

./batch-pair-windmill-vracer.sh
# ./eval-pair-windmill-vracer.sh
