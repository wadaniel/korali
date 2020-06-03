#!/bin/bash

# Brief: Quickly Re-run all example applications for basic sanity check.

###### Auxiliar Functions and Variables #########
 
source ../functions.sh

############# STEP 1 ##############

dir=$PWD/../../examples

logEcho "[Korali] Beginning examples test..."
logEcho "-------------------------------------"

pushd $dir >> $logFile 2>&1; check_result

./.run_test.sh >> $logFile 2>&1
check_result

popd >> $logFile 2>&1 
check_result

logEcho "-------------------------------------"
