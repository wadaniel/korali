#!/usr/bin/env bash

if [ $# -gt 0 ]; then
  cd $1
fi

##### Deleting Previous Results

echo "  + Deleting previous results..."
rm -rf _korali_result*

##### Recompiling C++

make clean
exit_code=$?

make -j4
exit_code=$(( $exit_code || $? ))

##### Running Tests

./run-cmaes
exit_code=$(( $exit_code || $? ))

./run-cmaes-direct
exit_code=$(( $exit_code || $? ))

./run-lmcma
exit_code=$(( $exit_code || $? ))

./run-lmcma-direct
exit_code=$(( $exit_code || $? ))

./run-tmcmc
exit_code=$(( $exit_code || $? ))


exit $exit_code
