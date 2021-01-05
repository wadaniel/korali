#!/bin/bash -l
#SBATCH --job-name="bPipeMaxMeanVel0096"
#SBATCH --output=bPipeMaxMeanVel0096_%j.out
#SBATCH --time=5:00:00
#SBATCH --nodes=96
#SBATCH --account=eth2
#SBATCH --partition=normal
#SBATCH --constraint=mc
#SBATCH --dependency=singleton

date 

cd ..

python3 run-korali.py -c 32 --ngens 200 --objective maxMeanVel --resultFolder _resultMaxMeanVel

date
