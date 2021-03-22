#!/bin/bash -l
#SBATCH --job-name="ABF2D_VRACER"
#SBATCH --output=ABF2D_VRACER_%j.out
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=eth2
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=mc

pushd ..

cat run-vracer.py

expDir=$SCRATCH/ABF2D_VRACER/$SLURM_JOB_ID
mkdir -p $expDir
cp run-vracer.py $expDir
cp -r _model $expDir

popd

pushd $expDir

OMP_NUM_THREADS=36 python3 run-vracer.py

popd

date
