#!/bin/bash

##############################################################################
# Brief: Test correct execution of solvers with non 0815 parametrization.
# Type: Regression Test 
# Description:
# This test run python scripts that set up a Korali problem  and run Korali 
# solvers. Solvers will be extensively configured to check for corner cases.
###############################################################################

source ../functions.sh

#################################################
# Execute Solver Scripts
#################################################

logEcho "[Korali] Beginning solver tests"

for file in *.py
do
  logEcho "-------------------------------------"
  logEcho "Running File: ${file%.*}"
  
  python3 ./$file >> $logFile 2>&1
  check_result

  log "[Korali] Removing results..."
  rm -rf "_korali_result" >> $logFile 2>&1
  check_result

  logEcho "-------------------------------------"
done

