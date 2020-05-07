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

cp run-alphabeta.py __test-alphabeta.py; check_result
cp run-budget.py    __test-budget.py; check_result 
cp run-ovens.py     __test-ovens.py; check_result   

##### If fast mode, then creating faster variations

if [[ $testMode == "fast" ]]; then

 echo "  + Creating fast mode test files..."
 
 sed -e 's%granularity = 0.%granularity = 0.4 #%g' run-alphabeta.py  > __test-alphabeta.py; check_result
 sed -e 's%granularity = 0.%granularity = 0.4 #%g' run-budget.py    > __test-budget.py; check_result
 sed -e 's%granularity = 1.%granularity = 20. #%g' run-ovens.py  > __test-ovens.py; check_result

fi

##### Running Tests

python3 ./__test-alphabeta.py; check_result
python3 ./__test-budget.py; check_result
python3 ./__test-ovens.py; check_result

##### Deleting Tests

rm __test-*; check_result
  
