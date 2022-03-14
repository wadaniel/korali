#! /usr/bin/env bash

if [ $# -lt 2 ] ; then
	echo "Usage: ./eval-vracer-swimmer.sh RUNNAME TASK"
	exit 1
fi

RUNNAME=$1
TASK=$2

# number of agents
NAGENTS=1

# number of nodes per worker
NRANKS=9

# number of cores per worker
NUMCORES=12

# Set number of nodes here
mpiflags="mpirun -n 2"

if [ ! -z $SLURM_NNODES ]; then
 N=$SLURM_NNODES
 mpiflags="srun -N $N -n $((N+1)) -c 12"
fi

RUNPATH="${SCRATCH}/korali/${RUNNAME}"
cp eval-vracer-swimmer ${RUNPATH}
cd ${RUNPATH}

set -ux

# $mpiflags ./eval-vracer-swimmer ${OPTIONS} -shapes "${OBJECTS}" -nAgents $NAGENTS
srun --nodes=$((N-1)) --ntasks-per-node=$NUMCORES --cpus-per-task=1 --threads-per-core=1 ./eval-vracer-swimmer -task $TASK -nAgents $NAGENTS -nRanks $(( $NRANKS * $NUMCORES )) : --nodes=1 --ntasks-per-node=1 --cpus-per-task=$NUMCORES --threads-per-core=1 ./eval-vracer-swimmer -task $TASK -nAgents $NAGENTS -nRanks $(( $NRANKS * $NUMCORES ))