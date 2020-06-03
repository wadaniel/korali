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

cp run-integration.py __test-integration.py; check_result

##### If fast mode, then creating faster variations

if [[ $testMode == "fast" ]]; then

 echo "  + Creating fast mode test files..."
 
 sed -e 's%\["Max Generations"\]%\["Max Generations"\] = 5 #%g' run-integration.py > __test-integration.py; check_result

fi

##### Running Tests

python3 ./__test-integration.py; check_result

##### Deleting Tests

rm __test-*; check_result
  
