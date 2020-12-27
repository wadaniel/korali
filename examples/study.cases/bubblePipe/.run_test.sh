#!/bin/bash

###### Auxiliar Functions and Variables #########

source ../../../tests/functions.sh

##### Deleting Previous Results

echo "  + Deleting previous results..." 
rm -rf _result*; check_result

##### Recompiling C++

make clean; check_result
make -j4 TEST=true; check_result

###### If this is macOS, C++ linking may not be automatic: do not run test
arch="$(uname -s)"
if [ "$arch" == "Darwin" ]; then
 log "[Korali] MacOS (Darwin) System Detected, aborting test."
 exit 0
fi

##### Running Tests

./run-korali; check_result
