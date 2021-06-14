#!/bin/bash -l
#SBATCH --job-name="fish64"
#SBATCH --output=output_fish64_%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=65
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=s929
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu

if [ $# -lt 1 ] ; then
	echo "Usage: ./fish64.sbatch RUNNAME"
	exit 1
fi
if [ $# -gt 0 ] ; then
	RUNNAME=$1
fi

RUNPATH="${SCRATCH}/CUP2D/${RUNNAME}"
mkdir -p ${RUNPATH}
cp ../run-vracer-fish ../batch-fish-vracer.sh ${FOLDERNAME}
cd ${FOLDERNAME}

./batch-fish-vracer.sh
