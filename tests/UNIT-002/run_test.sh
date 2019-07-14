#!/bin/bash

##############################################################################
# Brief: Distributed Linked Conduit for Distributed (MPI) Bayesian Inference
# Type: Unit Test 
# Description:
# Tests the distributed Linked conduit for a bayesian inference problem using
# a sequential heat diffusion solver on 2D. 
# Steps:
# 1 - Operation: Compile test case.
#     Expected result: Correct compilation with rc = 0.
#     If MPI is not installed (e.g., macOs) , it will not compile. 
# 2 - Operation: Run 8x1 distribution.
#     Expected result: 8 Concurrent teams of 1 MPI rank run with rc = 0.
#     If MPI is not installed (e.g., macOs) , it will not run.
# 3 - Operation: Run 4x2 distribution.
#     Expected result: 4 Concurrent teams of 2 MPI rank run with rc = 0.
#     If MPI is not installed (e.g., macOs) , it will not run.
# 4 - Operation: Run 1x8 distribution.
#     Expected result: 1 Teams of 8 MPI ranks run with rc = 0.
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

logEcho "[Korali] Compiling poisson_posterior..."
make clean >> $logFile 
check_result

make -j 4 >> $logFile 
check_result

############# STEP 2 ##############

logEcho "[Korali] Running mpirun -n 9 ./poisson_posterior 1..."
mpirun -n 9 ./poisson_posterior 1 >> $logFile
check_result

############# STEP 3 ##############

logEcho "[Korali] Running mpirun -n 9 ./poisson_posterior 4..."
mpirun -n 9 ./poisson_posterior 4 >> $logFile 
check_result

############# STEP 4 ##############

logEcho "[Korali] Running mpirun -n 9 ./poisson_posterior 8..."
mpirun -n 9 ./poisson_posterior 8 >> $logFile 
check_result
