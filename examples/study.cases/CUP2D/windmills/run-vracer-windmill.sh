#!/bin/bash

if [ $# -lt 1 ] ; then
	echo "Usage: ./run-vracer-windmill.sh RUNNAME"
	exit 1
fi
if [ $# -gt 0 ] ; then
	RUNNAME=$1
fi

# Set number of nodes here
mpiflags="mpirun -n 2"

if [ ! -z $SLURM_NNODES ]; then
 N=$SLURM_NNODES
 mpiflags="srun -N $N -n $((N+1)) -c 12"
fi

RUNPATH="${SCRATCH}/korali/${RUNNAME}"
mkdir -p ${RUNPATH}
cp run-vracer-windmill ${RUNPATH}
cp settings.sh ${RUNPATH}
cp profiles/freqnu2hz.dat ${RUNPATH}/profile.dat
cp profiles/freqnu2hz.dat ${RUNPATH} # indication of which data file to use
cd ${RUNPATH}

source settings.sh

set -ux

$mpiflags ./run-vracer-windmill ${OPTIONS} -shapes "${OBJECTS}"