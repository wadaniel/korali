#!/bin/bash

source ../functions.sh

constraints=(
"None"
"Inactive"
"Active at Max 1"
"Active at Max 2"
"Inactive at Max 1"
"Inactive at Max 1"
"Mixed"
"Stress"
)

#################################################
# CCMA-ES Termination Criterion Tests
#################################################

logEcho "[Korali] Beginning CMA-ES termination criterion tests"

for c in "${constraints[@]}"
do

  logEcho "-------------------------------------"
  logEcho "Testing Constraints: ${c}"
  logEcho "Running File: run-ccmaes.py"

  python3 ./run-ccmaes.py --constraint "${c}" >> $logFile 2>&1
  check_result

  log "[Korali] Removing results..."
  rm -rf "_korali_result" >> $logFile 2>&1
  check_result

  logEcho "-------------------------------------"

done
