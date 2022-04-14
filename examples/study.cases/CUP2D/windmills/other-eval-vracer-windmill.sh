#!/bin/bash

if [ $# -lt 1 ] ; then
	echo "Usage: ./eval-vracer-windmill.sh RUNNAME"
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
cp 4eval-vracer-windmill ${RUNPATH}
cp 4settings.sh ${RUNPATH}
cd ${RUNPATH}

source 4settings.sh

set -ux

$mpiflags ./4eval-vracer-windmill ${OPTIONS} -shapes "${OBJECTS}"