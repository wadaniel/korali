#!/usr/bin/env bash

##### Deleting Previous Results

echo "  + Deleting previous results..."
rm -rf _korali_result*
exit_code=$?

##### Running Tests

python3 ./run-cmaes.py
exit_code=$(( $exit_code || $? ))

python3 ./run-dea.py
exit_code=$(( $exit_code || $? ))

python3 ./run-lmcma.py
exit_code=$(( $exit_code || $? ))


exit $exit_code