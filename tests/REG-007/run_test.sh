#!/bin/bash

source ../functions.sh

#################################################
# Executor Test
#################################################

logEcho "[Korali] Beginning Executor test"

logEcho "-------------------------------------"
logEcho "Testing Executor: ${c}"
logEcho "Running File: run-execution.py"

python3 ./run-execution.py >> $logFile 2>&1
check_result

log "[Korali] Removing results..."
rm -rf "_korali_result" >> $logFile 2>&1
check_result

logEcho "-------------------------------------"
