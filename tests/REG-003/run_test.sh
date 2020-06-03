#!/bin/bash

# Brief: Re-run all example applications for basic sanity check.

###### Auxiliar Functions and Variables #########

source ../functions.sh

############# STEP 1 ##############

dir=$PWD/../../examples

echo "[Korali] Beginning examples test..."
echo "-------------------------------------"

pushd $dir; check_result

./.run_test.sh
check_result

popd
check_result

echo "-------------------------------------"
