#!/usr/bin/env bash

if [ $# -gt 0 ]; then
  cd $1
fi

##### Deleting Previous Results

echo "  + Deleting previous results..."
rm -rf _korali_result*
exit_code=$?

##### Running Test

python3 ./run-surrogates.py
exit_code=$(( $exit_code || $? ))


exit $exit_code