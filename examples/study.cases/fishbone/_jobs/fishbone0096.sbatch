#!/bin/bash -l
#SBATCH --job-name="fishbone0096"
#SBATCH --output=fish0096_%j.out
#SBATCH --time=5:00:00
#SBATCH --nodes=96
#SBATCH --account=eth2
#SBATCH --partition=normal
#SBATCH --constraint=mc
#SBATCH --dependency=singleton

date 

cd ..
cat run-korali.py
python3 run-korali.py -c 32 --ngens 3 --tmax 5.0

date
