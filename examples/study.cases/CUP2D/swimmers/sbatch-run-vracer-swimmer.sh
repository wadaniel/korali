#! /usr/bin/env bash

if [ $# -lt 1 ] ; then
	echo "Usage: ./sbatch-run-vracer-swimmer.sh RUNNAME"
	exit 1
fi
if [ $# -gt 0 ] ; then
	RUNNAME=$1
fi

# number of workers
# NWORKER=32
NWORKER=1

# number of nodes per worker
# NRANKS=8
NRANKS=9

# number of threads per worker
NUMTHREADS=12

# number of workers * number of nodes per worker
NNODES=$(( $NWORKER * $NRANKS ))

# setup run directory and copy necessary files
RUNPATH="${SCRATCH}/korali/${RUNNAME}"
mkdir -p ${RUNPATH}
cp run-vracer-swimmer ${RUNPATH}
cp settings.sh ${RUNPATH}
cd ${RUNPATH}

source settings.sh

cat <<EOF >daint_sbatch
#!/bin/bash -l
#SBATCH --job-name="${RUNNAME}"
#SBATCH --output=${RUNNAME}_out_%j.txt
#SBATCH --error=${RUNNAME}_err_%j.txt
# #SBATCH --time=24:00:00
# #SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --partition=debug
#SBATCH --nodes=$((NNODES+1))
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${NUMTHREADS}

#SBATCH --constraint=gpu
#SBATCH --account=s929

export OMP_NUM_THREADS=${NUMTHREADS}

srun ./run-vracer-swimmer ${OPTIONS} -shapes "${OBJECTS}" -nAgents $NAGENTS -nRanks $NRANKS

EOF

echo "Starting ${NWORKER} simulations each using ${NRANKS} ranks and ${NUMTHREADS} threads"
echo "----------------------------"

chmod 755 daint_sbatch
sbatch daint_sbatch