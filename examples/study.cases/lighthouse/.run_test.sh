#!/usr/bin/env bash

if [ $# -gt 0 ]; then
  cd $1
fi

##### Deleting Previous Results

echo "  + Deleting previous results..."
rm -rf _results_*
exit_code=$?

##### Running Tests

echo "  + Running test files..."

python3 ./run-example1.py
exit_code=$(( $exit_code || $? ))

python3 ./run-example2.py
exit_code=$(( $exit_code || $? ))


exit $exit_code