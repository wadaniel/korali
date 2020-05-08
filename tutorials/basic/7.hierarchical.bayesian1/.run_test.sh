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

cp phase0.py   __test-phase0.py; check_result
cp phase1.py   __test-phase1.py; check_result 
cp phase2.py   __test-phase2.py; check_result 
cp phase3a.py  __test-phase3a.py; check_result  
cp phase3b.py  __test-phase3b.py; check_result

##### If fast mode, then creating faster variations

if [[ $testMode == "fast" ]]; then

 echo "  + Creating fast mode test files..."
 
 sed -e 's%\["Population Size"\]%\["Population Size"\] = 300 #%g' phase1.py  > __test-phase1.py; check_result
 sed -e 's%\["Population Size"\]%\["Population Size"\] = 300 #%g' phase2.py  > __test-phase2.py; check_result
 sed -e 's%\["Population Size"\]%\["Population Size"\] = 300 #%g' phase3a.py  > __test-phase3a.py; check_result
 sed -e 's%\["Population Size"\]%\["Population Size"\] = 300 #%g' phase3b.py  > __test-phase3b.py; check_result

fi

##### Running Tests

python3 ./__test-phase0.py; check_result
python3 ./__test-phase1.py; check_result
python3 ./__test-phase2.py; check_result
python3 ./__test-phase3a.py; check_result
python3 ./__test-phase3b.py; check_result

##### Deleting Tests

rm __test-*; check_result
  
