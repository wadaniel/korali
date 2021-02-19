#!/usr/bin/env bash

if [ $# -gt 0 ]; then
  cd $1
fi

##### Deleting Previous Results

echo "  + Deleting previous results..."
rm -rf _result*
exit_code=$?


python3 ./run-cmaes.py
exit_code=$(( $exit_code || $? ))
python3 ./run-cmaes.py
exit_code=$(( $exit_code || $? ))

python3 ./run-gfpt.py
exit_code=$(( $exit_code || $? ))
python3 ./run-gfpt.py
exit_code=$(( $exit_code || $? ))

python3 ./run-sin.py
exit_code=$(( $exit_code || $? ))
python3 ./run-sin.py
exit_code=$(( $exit_code || $? ))

retuen $exit_code