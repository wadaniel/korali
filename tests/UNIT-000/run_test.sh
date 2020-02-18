#!/bin/bash

source ../functions.sh

pushd ../../tutorials/advanced/5.concurrent.execution/
check_result

############# STEP 1 ##############
logEcho "[Korali] Running run.py 1..."
./run.py 1 >> $logFile
check_result

############# STEP 2 ##############

logEcho "[Korali] Running run.py 4..."
./run.py 4 >> $logFile
check_result

############# STEP 3 ##############

logEcho "[Korali] Running run.py 8..."
./run.py 8 >> $logFile
check_result

rm -rf _korali_result >> $logFile 2>&1
check_result

popd
