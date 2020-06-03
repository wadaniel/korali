#!/bin/bash

source ../functions.sh

############# Preparing Test ##############

if [[ $MPICXX == "" ]]
then
 echo "[Korali] MPI not installed, skipping test."
 exit 0
fi

pushd ../../tutorials/advanced/6.running.mpi/
check_result
dir=$PWD

echo "-------------------------------------"
echo " Entering Folder: $dir"

############ STEP 1 #######

echo "[Korali] Compiling tests ..."
make clean >> $logFile 
check_result

make -j 4 >> $logFile 
check_result

for file in *.cpp
do
  if [ ! -f $file ]; then continue; fi
  
  execName=${file%.*}
  
  ############# STEP 2 ##############
  
  echo "[Korali] Running mpirun -n 9 ./$execName 1..."
  mpirun -n 9 ./$execName 1 >> $logFile
  check_result
  
  ############# STEP 3 ##############
  
  echo "[Korali] Running mpirun -n 9 ./$execName 4..."
  mpirun -n 9 ./$execName 4 >> $logFile 
  check_result
  
  ############# STEP 4 ##############
  
  echo "[Korali] Running mpirun -n 9 ./$execName 8..."
  mpirun -n 9 ./$execName 8 >> $logFile 
  check_result
done

echo "-------------------------------------"

popd
