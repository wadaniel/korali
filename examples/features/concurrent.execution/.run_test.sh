#!/usr/bin/env bash

if [ $# -gt 0 ]; then
  cd $1
fi

##### Deleting Previous Results

echo "  + Deleting previous results..."
rm -rf _korali_result*
exit_code=$?

##### Running Tests

python3 ./run.py 1
exit_code=$(( $exit_code || $? ))

python3 ./run.py 2
exit_code=$(( $exit_code || $? ))

python3 ./run.py 4
exit_code=$(( $exit_code || $? ))

python3 ./run.py 8
exit_code=$(( $exit_code || $? ))


retun $exit_code