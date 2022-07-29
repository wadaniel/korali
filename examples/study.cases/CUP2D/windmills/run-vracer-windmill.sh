#! /usr/bin/env bash

if [ $# -lt 1 ] ; then
	echo "Usage: ./run-vracer-windmill.sh RUNNAME"
	exit 1
fi
if [ $# -gt 0 ] ; then
	RUNNAME=$1
fi

# number of nodes per worker/simulation
NRANKS=1

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

# STATE:
# 1 = {omega1, omega2}
# 2 = velocity profile
# 3 = velocity profile + {omega1, omega2}

# REWARD:
# 1 = squared difference between deviation from the mean 
# 	  at time t and time t-1, normalized by the target profile
# 2 = squared difference between deviation from the mean 
# 	  at time t and time t-1, non-normalized
# 3 = log-likelihood for hypothetical normal distribution of profiles
# 4 = slight reward given, goal to teach agent to have angular velocity 
# 	  smaller than 10. 

# ALPHA:
# index of the simulation we wish to learn from,
# values between 0 and 10. 11 means we use all of them

# SEQLEN:
# the sequence length to be used for the RNNs

STATE=1
REWARD=5
ALPHA=6
SEQLEN=20

# source settings.sh

set -ux

# need at least two nodes 
# heterogenous run, that will have N-1 nodes with pure mpi, i.e. 12 ranks and 1 node with openMP, i.e. 1 rank and 12 threads
# srun --nodes=$((N-1)) --ntasks-per-node=$NUMCORES --cpus-per-task=1 --threads-per-core=1  ./run-vracer-windmill -state $(($STATE)) -reward $(($REWARD)) -alpha $(($ALPHA)) -seqLen $(($SEQLEN)) -nRanks $(( $NRANKS * $NUMCORES )) : --nodes=1 --ntasks-per-node=1 --cpus-per-task=$NUMCORES --threads-per-core=1 ./run-vracer-windmill -state $(($STATE)) -reward $(($REWARD)) -alpha $(($ALPHA)) -seqLen $(($SEQLEN)) -nRanks $(( $NRANKS * $NUMCORES ))

# now we use cuda to solve the poisson problem. 
# each simulation gets nranks 

srun --nodes=$((N-1)) --ntasks-per-node=1 --cpus-per-task=$NUMCORES --threads-per-core=1 ./run-vracer-windmill -state $(($STATE)) -reward $(($REWARD)) -alpha $(($ALPHA)) -seqLen $(($SEQLEN)) -nRanks $(( $NRANKS )) : --nodes=1 --ntasks-per-node=1 --cpus-per-task=$NUMCORES --threads-per-core=1 ./run-vracer-windmill -state $(($STATE)) -reward $(($REWARD)) -alpha $(($ALPHA)) -seqLen $(($SEQLEN)) -nRanks $(( $NRANKS ))
