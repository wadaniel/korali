#!/bin/bash -l
#SBATCH --job-name="fishbone0096"
#SBATCH --output=fish0096_%j.out
#SBATCH --time=12:00:00
#SBATCH --nodes=96
#SBATCH --account=eth2
#SBATCH --partition=normal
#SBATCH --constraint=mc

date 

cd ..
cat run-korali.py
python3 run-korali.py -r _result -c 32

date
