#!/bin/bash

##############################################################################
# Brief: Test for the Non-Intrusive Conduit 
# Type: Unit Test 
# Description:
# Simple test that runs 100 generations of CMAES running the ackley function
# that is executed as a standalone process
# Steps: 
# 1 - Operation: Run nonintrusive.py
#     Expected result: Runs 100 generations without errors and rc = 0.  
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

logEcho "[Korali] Running nonintrusive.py..."
./nonintrusive.py >> $logFile 2>&1
check_result