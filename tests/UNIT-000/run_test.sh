#!/bin/bash

##############################################################################
# Brief: Test for the External Conduit 
# Type: Unit Test 
# Description:
# Simple test that runs 30 generations of CMAES running the ackley function
# that is executed as a standalone process
# Steps: 
# 1 - Operation: Run external.py with 1 process
#     Expected result: Runs without errors and rc = 0.
# 3 - Operation: Run external.py with 4 concurrent processes
#     Expected result: Runs without errors and rc = 0.
# 4 - Operation: Run external.py with 8 concurrent processes
#     Expected result: Runs without errors and rc = 0.  
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

logEcho "[Korali] Running external.py 1..."
./external.py 1 >> $logFile 2>&1
check_result

############# STEP 2 ##############

logEcho "[Korali] Running external.py 4..."
./external.py 4 >> $logFile 2>&1
check_result

############# STEP 3 ##############

logEcho "[Korali] Running external.py 8..."
./external.py 8 >> $logFile 2>&1
check_result