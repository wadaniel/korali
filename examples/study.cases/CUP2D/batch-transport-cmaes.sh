#!/bin/bash

# Set number of nodes here
mpiflags="mpirun -n 2"

if [ ! -z $SLURM_NNODES ]; then
 N=$SLURM_NNODES
 mpiflags="srun -N $N -n $(($N+1)) -c 12"
fi

set -x

# AMR
$mpiflags ./run-cmaes-transport  -bpdx 32 -bpdy 32 -levelMax 3 -Rtol 0.1 -Ctol 0.01 -poissonType dirichlet -muteAll 1 -verbose 0 -tdump 0 -nu 0.000018 -tend 0 -shapes 'smartDisk_radius=.06'
