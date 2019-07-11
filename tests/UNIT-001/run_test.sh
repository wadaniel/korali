#!/bin/bash

##############################################################################
# Brief: Distributed Conduit for Sequential Bayesian Inference
# Type: Unit Test 
# Description:
# Tests the distributed for a bayesian inference problem using a sequential
# heat diffusion solver on 2D. 
# Steps: 
# 1 - Operation: Run 8x1 distribution.
#     Expected result: 8 Concurrent teams of 1 MPI rank run with rc = 0.
#     If MPI is not installed (e.g., macOs) , it will not run.
###############################################################################

###### Auxiliar Functions and Variables #########

curdir=$PWD
logFile=$curdir/test.log
echo "" > $logFile

function check_result()
{
 if [ ! $? -eq 0 ]
 then
  echo "[Korali] Error running test. Please check $logFile."
  exit -1
 fi 
}

# Logging and printing function.
function logEcho ()
{
 echo "$1"
 echo "$1" >> $logFile
}

############# STEP 1 ##############

if [[ $MPICXX == "" ]]
then
 echo "[Korali] MPI not installed, skipping test."
 exit 0
fi

logEcho "[Korali] Compiling heat2d_posterior..."
make -j 4 >> $logFile 2>&1
check_result

logEcho "[Korali] Running mpirun -n 8 ./heat2d_posterior..."
mpirun -n 9 ./heat2d_posterior >> $logFile 2>&1
check_result
