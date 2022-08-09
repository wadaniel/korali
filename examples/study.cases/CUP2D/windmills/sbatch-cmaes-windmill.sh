#! /usr/bin/env bash

if [ $# -lt 1 ] ; then
	echo "Usage: ./run-cmaes-windmill.sh RUNNAME"
	exit 1
fi
if [ $# -gt 0 ] ; then
	RUNNAME=$1
fi

# number of total nodes for all CUP simulations
N=16

# number of cores per nodes for worker/simulation
NUMCORES=12

RUNPATH="${SCRATCH}/korali/${RUNNAME}"
mkdir -p ${RUNPATH}
cp run-cmaes-windmill ${RUNPATH}
cp results/quickdiff/x_profile.dat ${RUNPATH}/x_profile.dat
cp results/quickdiff/y_profile.dat ${RUNPATH}/y_profile.dat
cd ${RUNPATH}

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


POP=$N
MU=$((N/2))
ALPHA=6
REWARD=1

cat <<EOF >daint_sbatch
#!/bin/bash -l
#SBATCH --account=s929
#SBATCH --constraint=gpu
#SBATCH --job-name="${RUNNAME}"
#SBATCH --output=${RUNNAME}_out_%j.txt
#SBATCH --error=${RUNNAME}_err_%j.txt
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --nodes=$((N+1))

srun --nodes=$(($POP)) --ntasks-per-node=1 --cpus-per-task=$NUMCORES --threads-per-core=1 ./run-cmaes-windmill -pop $(($POP)) -mu $(($MU)) -alpha $(($ALPHA)) -reward $(($REWARD)) : --nodes=1 --ntasks-per-node=1 --cpus-per-task=$NUMCORES --threads-per-core=1 ./run-cmaes-windmill -pop $(($POP)) -mu $(($MU)) -alpha $(($ALPHA)) -reward $(($REWARD))

EOF

# need at least two nodes 
# heterogenous run, that will have N-1 nodes with pure mpi, i.e. 12 ranks and 1 node with openMP, i.e. 1 rank and 12 threads
# srun --nodes=$((N-1)) --ntasks-per-node=$NUMCORES --cpus-per-task=1 --threads-per-core=1  ./run-vracer-windmill -state $(($STATE)) -reward $(($REWARD)) -alpha $(($ALPHA)) -seqLen $(($SEQLEN)) -nRanks $(( $NRANKS * $NUMCORES )) : --nodes=1 --ntasks-per-node=1 --cpus-per-task=$NUMCORES --threads-per-core=1 ./run-vracer-windmill -state $(($STATE)) -reward $(($REWARD)) -alpha $(($ALPHA)) -seqLen $(($SEQLEN)) -nRanks $(( $NRANKS * $NUMCORES ))

# now we use cuda to solve the poisson problem. 
# each simulation gets nranks 

# srun --nodes=$((N-1)) --ntasks-per-node=1 --cpus-per-task=$NUMCORES --threads-per-core=1 ./run-vracer-windmill -state $(($STATE)) -reward $(($REWARD)) -alpha $(($ALPHA)) -seqLen $(($SEQLEN)) -nRanks $(( $NRANKS )) : --nodes=1 --ntasks-per-node=1 --cpus-per-task=$NUMCORES --threads-per-core=1 ./run-vracer-windmill -state $(($STATE)) -reward $(($REWARD)) -alpha $(($ALPHA)) -seqLen $(($SEQLEN)) -nRanks $(( $NRANKS ))

chmod 755 daint_sbatch
sbatch daint_sbatch