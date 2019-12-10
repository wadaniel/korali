#!/bin/bash

source ../functions.sh

############# STEP 1 ##############

logEcho "[Korali] Running concurrent.py 1..."
./concurrent.py 1 >> $logFile
check_result

############# STEP 2 ##############

logEcho "[Korali] Running concurrent.py 4..."
./concurrent.py 4 >> $logFile
check_result

############# STEP 3 ##############

logEcho "[Korali] Running concurrent.py 8..."
./concurrent.py 8 >> $logFile
check_result

rm -rf _korali_result >> $logFile 2>&1
check_result
