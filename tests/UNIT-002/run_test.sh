#!/bin/bash

source ../functions.sh

############# Preparing Test ##############

if [[ $MPICXX == "" ]]
then
 echo "[Korali] MPI not installed, skipping test."
 exit 0
fi

pushd ../../tutorials/c2-distributed-mpi/
dir=$PWD

logEcho "-------------------------------------"
logEcho " Entering Folder: $dir"

############ STEP 1 #######

logEcho "[Korali] Compiling tests ..."
make clean >> $logFile 
check_result

make -j 4 >> $logFile 
check_result

for file in *.cpp
do
  if [ ! -f $file ]; then continue; fi
  
  execName=${file%.*}
  
  ############# STEP 2 ##############
  
  logEcho "[Korali] Running mpirun -n 9 ./$execName 1..."
  mpirun -n 9 ./$execName 1 >> $logFile
  check_result
  
  ############# STEP 3 ##############
  
  logEcho "[Korali] Running mpirun -n 9 ./$execName 4..."
  mpirun -n 9 ./$execName 4 >> $logFile 
  check_result
  
  ############# STEP 4 ##############
  
  logEcho "[Korali] Running mpirun -n 9 ./$execName 8..."
  mpirun -n 9 ./$execName 8 >> $logFile 
  check_result
done

logEcho "-------------------------------------"

popd
