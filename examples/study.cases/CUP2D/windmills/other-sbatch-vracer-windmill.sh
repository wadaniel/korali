#! /usr/bin/env bash

if [ $# -lt 1 ] ; then
	echo "Usage: ./sbatch-vracer-windmill.sh RUNNAME"
	exit 1
fi
if [ $# -gt 0 ] ; then
	RUNNAME=$1
fi

# number of workers/simulations in parallel
NWORKER=16

# number of nodes per worker/simulation
NRANKS=1

# number of cores per nodes (for workers)
NUMCORES=12

# number of worker * number of nodes per worker = number of nodes in total
NNODES=$(( $NWORKER * $NRANKS))

# setup run directory and copy necessary files
RUNPATH="${SCRATCH}/korali/${RUNNAME}"
mkdir -p ${RUNPATH}
cp other-run-vracer-windmill ${RUNPATH}
cp settings.sh ${RUNPATH}
cp avgprofiles/avgprofiles.dat ${RUNPATH}/
# cp avgprofiles/avgprofile_*.dat ${RUNPATH}/
# cp avgprofiles/stdprofile_*.dat ${RUNPATH}/
cd ${RUNPATH}

source settings.sh

cat <<EOF >daint_sbatch
#!/bin/bash -l
#SBATCH --account=s929
#SBATCH --constraint=gpu
#SBATCH --job-name="${RUNNAME}"
#SBATCH --output=${RUNNAME}_out_%j.txt
#SBATCH --error=${RUNNAME}_err_%j.txt
#SBATCH --time=1:00:00
#SBATCH --partition=normal
#SBATCH --nodes=$((NNODES+1))

srun --nodes=$NNODES --ntasks-per-node=1 --cpus-per-task=$NUMCORES --threads-per-core=1 ./other-run-vracer-windmill ${OPTIONS} -shapes "${OBJECTS}" -nRanks $(( $NRANKS )) : --nodes=1 --ntasks-per-node=1 --cpus-per-task=$NUMCORES --threads-per-core=1 ./other-run-vracer-windmill ${OPTIONS} -shapes "${OBJECTS}" -nRanks $(( $NRANKS ))
EOF

echo "Starting ${NWORKER} simulations each using ${NRANKS} nodes with ${NUMCORES} cores"
echo "----------------------------"

chmod 755 daint_sbatch
sbatch daint_sbatch


# srun --nodes=$NNODES --ntasks-per-node=$NUMCORES --cpus-per-task=1 --threads-per-core=1  ./run-vracer-windmill ${OPTIONS} -shapes "${OBJECTS}" -nRanks $(( $NRANKS * $NUMCORES )) : --nodes=1 --ntasks-per-node=1 --cpus-per-task=$NUMCORES --threads-per-core=1 ./run-vracer-windmill -nRanks $(( $NRANKS * $NUMCORES ))
# srun ./run-vracer-windmill ${OPTIONS} -shapes "${OBJECTS}"
# srun ./eval-vracer-windmill ${OPTIONS} -shapes "${OBJECTS}"