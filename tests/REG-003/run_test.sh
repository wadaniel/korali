#!/bin/bash

###### Auxiliar Functions and Variables #########

source ../functions.sh

############# STEP 1 ##############

pushd ../../docs
check_result

# Run documentation builder
./build.sh
check_result

popd
check_result
