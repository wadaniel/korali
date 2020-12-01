#!/bin/bash

###### Auxiliar Functions and Variables #########

source ../../../tests/functions.sh

##### Deleting Previous Results

echo "  + Deleting previous results..." 
rm -rf _korali_result*; check_result

##### Recompiling C++

make clean; check_result
make -j4; check_result

###### If this is macOS, C++ linking may not be automatic: do not run test
arch="$(uname -s)"
if [ "$arch" == "Darwin" ]; then
 log "[Korali] MacOS (Darwin) System Detected, aborting test."
 exit 0
fi

##### Running Tests

./run-cmaes; check_result
./run-cmaes-direct; check_result
./run-lmcma; check_result
./run-lmcma-direct; check_result
./run-tmcmc; check_result

