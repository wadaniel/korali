#! /usr/bin/env bash

if [ $# -lt 1 ] ; then
	echo "Usage: ./run-vracer-windmill.sh RUNNAME"
	exit 1
fi
if [ $# -gt 0 ] ; then
	RUNNAME=$1
fi

# number of nodes per worker/simulation
NRANKS=2

# number of cores per nodes for worker/simulation
NUMCORES=12

# Set number of nodes here
mpiflags="mpirun -n 12"

if [ ! -z $SLURM_NNODES ]; then
 N=$SLURM_NNODES
fi

RUNPATH="${SCRATCH}/korali/${RUNNAME}"
mkdir -p ${RUNPATH}
cp run-vracer-windmill ${RUNPATH}
# cp settings.sh ${RUNPATH}
cp avgprofiles/avgprofiles.dat ${RUNPATH}/avgprofiles.dat
cp avgprofiles/stdprofiles.dat ${RUNPATH}/stdprofiles.dat
cd ${RUNPATH}

STATE=1
REWARD=1
ALPHA=6
SEQLEN=40

# source settings.sh

set -ux

# need at least two nodes 
# heterogenous run, that will have N-1 nodes with pure mpi, i.e. 12 ranks and 1 node with openMP, i.e. 1 rank and 12 threads
srun --nodes=$((N-1)) --ntasks-per-node=$NUMCORES --cpus-per-task=1 --threads-per-core=1  ./run-vracer-windmill -state $(($STATE)) -reward $(($REWARD)) -alpha $(($ALPHA)) -seqLen $(($SEQLEN)) -nRanks $(( $NRANKS * $NUMCORES )) : --nodes=1 --ntasks-per-node=1 --cpus-per-task=$NUMCORES --threads-per-core=1 ./run-vracer-windmill -state $(($STATE)) -reward $(($REWARD)) -alpha $(($ALPHA)) -seqLen $(($SEQLEN)) -nRanks $(( $NRANKS * $NUMCORES ))

# now we use cuda to solve the poisson problem. 
# each simulation gets nranks 

# srun --nodes=$((N-1)) --ntasks-per-node=1 --cpus-per-task=$NUMCORES --threads-per-core=1 ./run-vracer-windmill -state $(($STATE)) -reward $(($REWARD)) -alpha $(($ALPHA)) -seqLen $(($SEQLEN)) -nRanks $(( $NRANKS )) : --nodes=1 --ntasks-per-node=1 --cpus-per-task=$NUMCORES --threads-per-core=1 ./run-vracer-windmill -state $(($STATE)) -reward $(($REWARD)) -alpha $(($ALPHA)) -seqLen $(($SEQLEN)) -nRanks $(( $NRANKS ))
