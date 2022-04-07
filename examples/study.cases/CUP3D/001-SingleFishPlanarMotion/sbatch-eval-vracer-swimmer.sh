#! /usr/bin/env bash

if [ $# -lt 3 ] ; then
	echo "Usage: ./sbatch-eval-vracer-swimmer.sh RUNNAME TASK DIMENSION"
	exit 1
fi

RUNNAME=$1
TASK=$2
DIMENSION=$3
EXECUTABLE=eval-vracer-swimmer-"${DIMENSION}"D

EXTRA=
if [ $DIMENSION == 3 ]
then
source _model3D/settings3D.sh
echo ${FACTORY}
EXTRA=${OPTIONS}" -factory-content \""${FACTORY}"\""
fi
echo ${EXTRA}

NAGENTS=1   # number of agents
NWORKER=3   # number of workers
NRANKS=1    # nodes per worker
NUMCORES=12 # cores per worker
NNODES=$(( $NWORKER * $NRANKS )) # number of workers * number of nodes per worker

# setup run directory and copy necessary files
RUNPATH="${SCRATCH}/korali/${RUNNAME}"
mkdir -p ${RUNPATH}
cp $EXECUTABLE ${RUNPATH}
cd ${RUNPATH}

cat <<EOF >daint_sbatch_testing-"${DIMENSION}"D
#!/bin/bash -l
#SBATCH --account=s929
#SBATCH --constraint=gpu
#SBATCH --job-name="${RUNNAME}"
#SBATCH --time=00:30:00
#SBATCH --partition=debug
#SBATCH --nodes=$((NNODES+1))

srun --nodes=$NNODES --ntasks-per-node=$NUMCORES --cpus-per-task=1 --threads-per-core=1 ./$EXECUTABLE $EXTRA -task $TASK -nAgents $NAGENTS -nRanks $(( $NRANKS * $NUMCORES )) : --nodes=1 --ntasks-per-node=1 --cpus-per-task=$NUMCORES --threads-per-core=1 ./$EXECUTABLE $EXTRA -task $TASK -nAgents $NAGENTS -nRanks $(( $NRANKS * $NUMCORES ))
EOF

chmod 755 daint_sbatch_testing-"${DIMENSION}"D
sbatch daint_sbatch_testing-"${DIMENSION}"D
