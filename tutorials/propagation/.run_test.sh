#!/bin/bash

###### Auxiliar Functions and Variables #########

source ../../../tests/functions.sh

##### Getting Test Type

getTestMode $1

##### Deleting Previous Results

echo "  + Deleting previous results..."
rm -rf _korali_result* _executor_output; check_result

##### Creating test files

echo "  + Creating test files..."

cp run-execution.py __test-execution.py; check_result

##### If fast mode, then creating faster variations

if [[ $testMode == "fast" ]]; then

 echo "  + Creating fast mode test files..."
 
 sed -e 's%k.run(e)%e["Solver"]["Termination Criteria"]["Max Model Evaluations"] = 5; k.run(e) #%g' run-execution.py  > __test-execution.py; check_result

fi

##### Running Tests

python3 ./__test-execution.py; check_result

##### Deleting Tests

rm __test-*; check_result
  
