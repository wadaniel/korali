#!/bin/bash

source ../functions.sh

pushd ../../tutorials/advanced/5.concurrent.execution/
check_result

############# STEP 1 ##############
echo "[Korali] Running run.py 1..."
./run.py 1 >> $logFile
check_result

############# STEP 2 ##############

echo "[Korali] Running run.py 4..."
./run.py 4 >> $logFile
check_result

############# STEP 3 ##############

echo "[Korali] Running run.py 8..."
./run.py 8 >> $logFile
check_result

rm -rf _korali_result
check_result

popd
