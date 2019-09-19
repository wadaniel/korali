#!/bin/bash

source ../functions.sh

############# STEP 1 ##############

logEcho "[Korali] Running external.py 1..."
./external.py 1 >> $logFile
check_result

############# STEP 2 ##############

logEcho "[Korali] Running external.py 4..."
./external.py 4 >> $logFile
check_result

############# STEP 3 ##############

logEcho "[Korali] Running external.py 8..."
./external.py 8 >> $logFile
check_result

rm -rf _korali_result >> $logFile 2>&1
check_result
