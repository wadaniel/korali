#!/bin/bash

###### Auxiliar Functions and Variables #########

source ../../../../tests/functions.sh

###### Getting MNIST Data
./get_data.sh

##### Deleting Previous Results 

echo "  + Deleting previous results..."
rm -rf _korali_result*; check_result

##### Creating test files
 
echo "  + Creating test files..."

sed -e 's%epochs = %epochs = 1#%g' \
        run-mnist.py > __test-mnist.py; check_result

##### Running Test

OMP_NUM_THREADS=4 python3 ./__test-mnist.py; check_result

##### Deleting Tests

rm -rf __test-*; check_result
  
  
