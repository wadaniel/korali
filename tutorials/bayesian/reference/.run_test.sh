#!/bin/bash

###### Auxiliar Functions and Variables #########

source ../../../tests/functions.sh

##### Getting Test Type

getTestMode $1

##### Deleting Previous Results

echo "  + Deleting previous results..."
rm -rf _korali_result*; check_result

##### Creating test files

echo "  + Creating test files..."

cp run-cmaes.py  __test-cmaes.py; check_result 
cp run-nested.py __test-nested.py; check_result
cp run-tmcmc.py __test-tmcmc.py ; check_result
cp run-mtmcmc.py __test-mtmcmc.py ; check_result

##### If fast mode, then creating faster variations

if [[ $testMode == "fast" ]]; then

 echo "  + Creating fast mode test files..."
 
 sed -e 's%\["Max Generations"\]%\["Max Generations"\] = 5 #%g'   run-cmaes.py  > __test-cmaes.py; check_result
 sed -e 's%\["Max Generations"\]%\["Max Generations"\] = 1000 #%g' run-nested.py > __test-nested.py; check_result
 sed -e 's%\["Population Size"\]%\["Population Size"\] = 100 #%g' run-tmcmc.py  > __test-tmcmc.py; check_result
 sed -e 's%\["Population Size"\]%\["Population Size"\] = 100 #%g' run-mtmcmc.py > __test-mtmcmc.py; check_result

fi

##### Running Tests

python3 ./__test-cmaes.py; check_result
python3 ./__test-nested.py; check_result
python3 ./__test-tmcmc.py; check_result
python3 ./__test-mtmcmc.py; check_result

##### Deleting Tests

rm __test-*; check_result
  
