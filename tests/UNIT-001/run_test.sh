#!/bin/bash

##############################################################################
# Brief: Test for the Distributed
# Type: Unit Test 
# Description:
# Tests several configurations of the distributed conduit on a heat2d posterior
# problem that employs an MPI solver.
# Steps: 
# 1 - Operation: Run 8x1 distribution.
#     Expected result: 8 Concurrent teams of 1 MPI rank run with rc = 0.
#     If MPI is not installed (e.g., macOs) , it will not run.
# 2 - Operation: Run 4x2 distribution.
#     Expected result: 4 Concurrent teams of 2 MPI ranks run with rc = 0.
#     If MPI is not installed (e.g., macOs) , it will not run.
# 3 - Operation: Run 1x8 distribution.
#     Expected result: 1 Team of 8 MPI ranks run with rc = 0.  
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

logEcho "[Korali] Running mpirun -n 8 ./heat2d_posterior..."
mpirun -n 8 ./heat2d_posterior >> $logFile 2>&1
check_result
