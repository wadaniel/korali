#!/bin/bash

###### Auxiliar Functions and Variables #########

source ../../../tests/functions.sh

##### Deleting Previous Results

echo "  + Deleting previous results..."
rm -rf _korali_result*; check_result

##### Running Tests

python3 ./run-saem-population-simple.py; check_result
python3 ./run-hsaem-population-simple.py; check_result
python3 ./run-hsaem-n-d.py; check_result
## Only run one of each, because logistic and normal examples are slow:
python3 ./run-hsaem-logistic.py; check_result
#python3 ./run-hsaem-normal.py; check_result
#python3 ./run-hsaem-logistic-custom.py; check_result
python3 ./run-hsaem-normal-custom.py; check_result

  
