#!/bin/bash -l
#SBATCH --job-name="ABF3D_VRACER"
#SBATCH --output=ABF3D_VRACER_%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --account=eth2
#SBATCH --partition=normal
#SBATCH --constraint=mc

date

pushd ..

cat run-vracer.cpp

expDir=$SCRATCH/ABF3D_VRACER/$SLURM_JOB_ID
mkdir -p $expDir
cp run-vracer $expDir
cp -r _config $expDir

popd

pushd $expDir

OMP_NUM_THREADS=36 ./run-vracer

popd

date
