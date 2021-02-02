#!/bin/bash -l
#SBATCH --job-name="ABF2D_GFPT"
#SBATCH --output=ABF2D_GFPT_%j.out
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=eth2
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=mc

pushd ..

cat run-gfpt.py

expDir=$SCRATCH/ABF2D_GFPT/$SLURM_JOB_ID
mkdir -p $expDir
cp run-gfpt.py $expDir
cp -r _model $expDir

popd

pushd $expDir

OMP_NUM_THREADS=36 python3 run-gfpt.py

popd

date
