#!/bin/bash

# Check Code Formatting

###### Auxiliar Functions and Variables #########

source ../functions.sh

############# STEP 1 ##############


logEcho "[Korali] Beginning C++ code formatting check..."

pushd ../../tools/dev-tools/
check_result

 ./check_style_cxx.sh >> $logFile 2>&1
 check_result

popd
check_result
