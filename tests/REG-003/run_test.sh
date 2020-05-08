#!/bin/bash

# Brief: Quickly Re-run all Python example applications for basic sanity check.

###### Auxiliar Functions and Variables #########

source ../functions.sh

############# STEP 1 ##############

pushd ../../tutorials

logEcho "[Korali] Beginning tutorial tests..."

for dir in ./basic/*/
do
  logEcho "-------------------------------------"
  logEcho " Entering Folder: $dir"

  pushd $dir >> $logFile 2>&1
  check_result

  logEcho " Runnning full test script..."
  ./.run_test.sh --full >> $logFile 2>&1
  check_result

  popd >> $logFile 2>&1
  check_result
  logEcho "-------------------------------------"
done

popd
