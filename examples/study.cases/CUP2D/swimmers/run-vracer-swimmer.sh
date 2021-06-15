#!/bin/bash

# Set number of nodes here
mpiflags="mpirun -n 2"

if [ ! -z $SLURM_NNODES ]; then
 N=$SLURM_NNODES
 mpiflags="srun -N $N -n $((N+1))  -c 12"
fi

set -ux

source settings.sh

$mpiflags ./run-vracer-swimmer ${OPTIONS} -shapes ${OBJECTS}
