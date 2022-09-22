#!/bin/bash

###### Auxiliar Functions and Variables #########

source ../../../../tests/functions.sh

##### Deleting Previous Results

echo "  + Deleting previous results..."
rm -rf _korali_result*; check_result

##### Creating test files

echo "  + Creating test files..."

rm -rf __test-*; check_result

for file in *.py
do
 sed -e 's%Defining Termination Criteria%Defining Termination Criteria\ne["Solver"]["Termination Criteria"]["Max Generations"] = 20\n%g' \
        ${file} > __test-${file}; check_result
done

##### Running Test

for file in __test-*.py
do
 echo "Running ${file} ..."
 OMP_NUM_THREADS=4 python3 ${file}; check_result
done

##### Deleting Tests

rm -rf __test-*; check_result
