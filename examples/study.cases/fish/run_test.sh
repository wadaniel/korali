#!/bin/bash

# Set number of nodes here
N=$SLURM_NNODES

rm -r _results
OMP_NUM_THREADS=12 srun -N $N -n $N -c 12 ./run-korali -poissonType cosine -muteAll 1 -bpdx 32 -bpdy 16 -tdump 0 -nu 0.000018 -tend 0 -shapes 'halfDisk_radius=.06_angle=20_xpos=.2_bForced=1_bFixed=1_xvel=0.15_tAccel=5,stefanfish_L=.2_xpos=.5'
