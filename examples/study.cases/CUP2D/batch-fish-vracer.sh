#!/bin/bash

# Set number of nodes here
mpiflags="mpirun -n 2"

if [ ! -z $SLURM_NNODES ]; then
 N=$SLURM_NNODES
 mpiflags="srun -N $N -n $N  -c 12"
fi

set -x

# AMR
$mpiflags ./run-vracer-fish  -bpdx 16 -bpdy 8 -levelMax 5 -Rtol 0.1 -Ctol 0.01 -CFL 0.2 -muteAll 1 -verbose 0 -tdump 0 -nu 0.00018 -tend 0 -shapes 'halfDisk_radius=.06_angle=20_xpos=.2_bForced=1_bFixed=1_xvel=0.15_tAccel=5,stefanfish_L=0.2_xpos=.4_bFixedy=1'
