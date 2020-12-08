#!/bin/bash

###### Auxiliar Functions and Variables #########

source ../../../tests/functions.sh

##### Deleting Previous Results

echo "  + Deleting previous results..." 
rm -rf _result*; check_result

##### Running Tests

python3 ./run-cmaes.py; check_result
python3 ./run-cmaes.py; check_result

###### Check if necessary python modules are installed ######
python3 -m pip show scipy
if [ $? -ne 0 ]; then
 echo "[Korali] Scipy not found, aborting test"
 exit 0
fi

python3 ./run-gfpt.py; check_result
python3 ./run-gfpt.py; check_result

