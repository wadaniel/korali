#!/usr/bin/env bash

##### Deleting Previous Results

echo "  + Deleting previous results..."
rm -rf _korali_result* _executor_output
exit_code=$?

##### Running Tests

python3 ./run-execution.py
exit_code=$(( $exit_code || $? ))

python3 ./run-uncertainty-propagation.py
exit_code=$(( $exit_code || $? ))


exit $exit_code