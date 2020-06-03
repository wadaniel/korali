#!/bin/bash

###### Auxiliar Functions and Variables #########

source ../../../tests/functions.sh

##### Deleting Previous Results

echo "  + Deleting previous results..."
./_setup/clean.sh; check_result

##### Running Tests

python3 ./run-phase0.py; check_result
python3 ./run-phase1.py; check_result
python3 ./run-phase2.py; check_result
python3 ./run-phase3a.py; check_result
python3 ./run-phase3b.py; check_result

