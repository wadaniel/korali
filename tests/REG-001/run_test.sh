#!/bin/bash

# Check Code Formatting

###### Auxiliar Functions and Variables #########
 
source ../functions.sh

############# STEP 1 ##############

echo "[Korali] Beginning code formatting check..."

pushd ../../tools/dev-tools/
check_result

 ./check_style_cxx.sh 
 check_result

popd
check_result
