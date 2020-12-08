#!/bin/bash

###### Auxiliar Functions and Variables #########

source ../../../tests/functions.sh

##### Deleting Previous Results

echo "  + Deleting previous results..." 
rm -rf _result*; check_result

##### Running Tests

python3 ./run-cmaes.py; check_result
python3 ./run-cmaes.py; check_result
python3 ./run-gfpt.py; check_result
python3 ./run-gfpt.py; check_result

