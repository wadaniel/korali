#!/usr/bin/env bash

##### Deleting Previous Results

echo "[Korali] + Deleting previous results..."
rm -rf _korali_result*
exit_code=$?

##### Running Tests

python3 ./run-saem-gaussian-mixture.py
exit_code=$(( $exit_code || $? ))

python3 ./run-saem.py
exit_code=$(( $exit_code || $? ))

return $exit_code


