#!/bin/bash

# Set number of nodes here
mpiflags="mpirun -n 2"

if [ ! -z $SLURM_NNODES ]; then
 N=$SLURM_NNODES
 mpiflags="srun -N $N -n $(($N+1)) -c 12"
fi

set -x

# AMR
$mpiflags ./run-vracer  -bpdx 32 -bpdy 16 -levelMax 3 -Rtol 0.1 -Ctol 0.01 -poissonType cosine -muteAll 1 -verbose 0 -tdump 0 -nu 0.000018 -tend 0 -poissonType cosine -shapes 'smartDisk_radius=.06_xpos=.2'

# NO AMR
# $mpiflags ./run-vracer -bpdx 32 -bpdy 16 -poissonType cosine -muteAll 1 -tdump 0 -nu 0.000018 -tend 0 -shapes 'halfDisk_radius=.06_angle=20_xpos=.2_bForced=1_bFixed=1_xvel=0.15_tAccel=5,stefanfish_L=.2_xpos=.5' 
