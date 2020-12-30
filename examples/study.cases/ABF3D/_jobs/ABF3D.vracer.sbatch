#!/bin/bash -l
#SBATCH --job-name="ABF3D_VRACER"
#SBATCH --output=/scratch/snx3000/smartin/ABF3D_VRACER/log%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=eth2
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=mc

date

pushd ..

cat run-vracer.cpp

expDir=$SCRATCH/ABF3D_VRACER/$SLURM_JOB_ID
mkdir $expDir
cp run-vracer $expDir
cp -r _config $expDir

popd

pushd $expDir

OMP_NUM_THREADS=36 ./run-vracer

popd

date
