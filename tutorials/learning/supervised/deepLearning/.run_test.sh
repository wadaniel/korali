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

sed -e 's%\plt.%#plt.%g' \
        run-sin.py > __test-sin.py; check_result

##### If fast mode, then creating faster variations

if [[ $testMode == "fast" ]]; then

 echo "  + Creating fast mode test files..."
 
sed -e 's%\["Max Generations"\]%\["Max Generations"\] = 100 #%g' \
    -e 's%\plt.%#plt.%g' \
        run-sin.py > __test-sin.py; check_result

fi

##### Running Tests

OMP_NUM_THREADS=1 python3 ./__test-sin.py; check_result

##### Deleting Tests

rm -rf __test-*; check_result
  
