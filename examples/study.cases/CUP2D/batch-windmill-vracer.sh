#!/bin/bash

# Set number of nodes here
mpiflags="mpirun -n 2"

if [ ! -z $SLURM_NNODES ]; then
 N=$SLURM_NNODES
 mpiflags="srun -N $N -n $(($N+1)) -c 12"
fi

set -x

# Defaults for Options
BPDX=${BPDX:-8} # number of blocks in x direction, 8 cells per block in compilation
BPDY=${BPDY:-8}
LEVELS=${LEVELS:-5} # each block can be refined twice, double number of points per refinement. 
RTOL=${RTOL-0.1}
CTOL=${CTOL-0.01}
EXTENT=${EXTENT:-1} # direction with most grid points is of size 1, (0, 0) point in bottom left
CFL=${CFL:-0.5}

# XPOS=${XPOS:-0.2}

# YPOS=${YPOS:-0.2}
# YPOS2=${YPOS2:-0.4}
# YPOS3=${YPOS3:-0.6}
# YPOS4=${YPOS4:-0.8}

# XVEL=${XVEL:-0.15}

# MAAXIS=${MAAXIS:-0.0375}
# MIAXIS=${MIAXIS:-0.01}

NU=${NU:-0.0001125}

# OBJECTS="windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS bForced=1 bFixed=1 xvel=$XVEL tAccel=0 bBlockAng=0
# windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS2 bForced=1 xvel=$XVEL tAccel=0 bBlockAng=0
# windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS3 bForced=1 xvel=$XVEL tAccel=0 bBlockAng=0
# windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS4 bForced=1 xvel=$XVEL tAccel=0 bBlockAng=0
# "

$mpiflags ./run-vracer-windmill -bpdx $BPDX -bdpy $BPDY -levelMax $LEVELS -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -tdump 0.1 -nu $NU -tend 50 -poissonType dirichlet -iterativePensalization 0 -muteAll 0 -verbose 1 "windmill_semiAxisX=0.0375_semiAxisY=0.01_xpos=0.2_ypos=0.2_bForced=1_bFixed=1_xvel=0.15_tAccel=0_bBlockAng=0,windmill_semiAxisX=0.0375_semiAxisY=0.01_xpos=0.2_ypos=0.4_bForced=1_bFixed=1_xvel=0.15_tAccel=0_bBlockAng=0,windmill_semiAxisX=0.0375_semiAxisY=0.01_xpos=0.2_ypos=0.6_bForced=1_bFixed=1_xvel=0.15_tAccel=0_bBlockAng=0,windmill_semiAxisX=0.0375_semiAxisY=0.01_xpos=0.2_ypos=0.8_bForced=1_bFixed=1_xvel=0.15_tAccel=0_bBlockAng=0,"

# AMR
#$mpiflags ./run-vracer-transport  -bpdx 32 -bpdy 32 -levelMax 3 -Rtol 0.1 -Ctol 0.01 -poissonType dirichlet -muteAll 1 -verbose 0 -tdump 0 -nu 0.000018 -tend 0 -shapes ''

# NO AMR
# $mpiflags ./run-vracer -bpdx 32 -bpdy 16 -poissonType cosine -muteAll 1 -tdump 0 -nu 0.000018 -tend 0 -shapes 'halfDisk_radius=.06_angle=20_xpos=.2_bForced=1_bFixed=1_xvel=0.15_tAccel=5,stefanfish_L=.2_xpos=.5' 
