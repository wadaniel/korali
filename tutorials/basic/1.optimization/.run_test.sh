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
cp run-dea.py    __test-dea.py; check_result 
cp run-lmcma.py  __test-lmcma.py; check_result 
cp run-rprop.py  __test-rprop.py; check_result  

##### If fast mode, then creating faster variations

if [[ $testMode == "fast" ]]; then

 echo "  + Creating fast mode test files..."
 
 sed -e 's%\["Max Generations"\]%\["Max Generations"\] = 5 #%g' run-cmaes.py  > __test-cmaes.py; check_result
 sed -e 's%\["Max Generations"\]%\["Max Generations"\] = 5 #%g' run-dea.py    > __test-dea.py; check_result
 sed -e 's%\["Max Generations"\]%\["Max Generations"\] = 5 #%g' run-lmcma.py  > __test-lmcma.py; check_result
 sed -e 's%\["Max Generations"\]%\["Max Generations"\] = 5 #%g' run-rprop.py  > __test-rprop.py; check_result

fi

##### Running Tests

python3 ./__test-cmaes.py; check_result
python3 ./__test-dea.py; check_result
python3 ./__test-lmcma.py; check_result
python3 ./__test-rprop.py; check_result

##### Deleting Tests

rm __test-*; check_result
  
