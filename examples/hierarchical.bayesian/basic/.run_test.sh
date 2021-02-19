#!/usr/bin/env bash

if [ $# -gt 0 ]; then
  cd $1
fi

##### Deleting Previous Results

echo "  + Deleting previous results..."

pushd _setup
exit_code=$?

./clean.sh
exit_code=$(( $exit_code || $? ))

popd
exit_code=$(( $exit_code || $? ))


##### Running Tests

python3 ./phase0.py
exit_code=$(( $exit_code || $? ))

python3 ./phase1.py
exit_code=$(( $exit_code || $? ))

python3 ./phase2.py
exit_code=$(( $exit_code || $? ))

python3 ./phase3a.py
exit_code=$(( $exit_code || $? ))

python3 ./phase3b.py
exit_code=$(( $exit_code || $? ))

exit $exit_code