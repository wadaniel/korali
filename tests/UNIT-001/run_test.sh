#!/bin/bash

##############################################################################
# Brief: Distributed Linked Conduit for Sequential Bayesian Inference
# Type: Unit Test 
# Description:
# Tests the distributed linked conduit for a bayesian inference problem using
# a sequential heat diffusion solver on 2D. 
# Steps: 
# 1 - Operation: Compile test case.
#     Expected result: Correct compilation with rc = 0.
#     If MPI is not installed (e.g., macOs) , it will not compile. 
# 2 - Operation: Run 8x1 distribution.
#     Expected result: 8 Concurrent teams of 1 MPI rank run with rc = 0.
#     If MPI is not installed (e.g., macOs) , it will not run.
###############################################################################

###### Auxiliar Functions and Variables #########

source ../functions.sh

############# STEP 1 ##############

if [[ $MPICXX == "" ]]
then
 echo "[Korali] MPI not installed, skipping test."
 exit 0
fi

logEcho "[Korali] Compiling heat2d_posterior..."

make clean >> $logFile 
check_result

make -j 4 >> $logFile
check_result

############# STEP 2 ##############

logEcho "[Korali] Running mpirun -n 9 ./heat2d_posterior..."
mpirun -n 9 ./heat2d_posterior >> $logFile
check_result
