#!/bin/bash

##############################################################################
# Brief: Quickly Re-run all Python example applications for basic sanity check.
# Type: Regression Test
# Description:
# This test finds and runs A Tutorials /tutorials folder to make sure
# the typical use cases still work.
# Steps:
# 1 - Operation: List and run all .py scripts in the tutorials/python folder.
#     Expected result: all of the .py scripts will run for less than 20 secs
#     and rc = 0.
###############################################################################

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

  logEcho "----------------------------------------------------------"
  logEcho " Run test script."
  ./.run_test.sh --fast >> $logFile 2>&1
  check_result

  popd >> $logFile 2>&1
  check_result
  logEcho "-------------------------------------"
done

popd
