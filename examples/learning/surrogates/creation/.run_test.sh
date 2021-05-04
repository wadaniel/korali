#!/bin/bash

###### Auxiliar Functions and Variables #########

source ../../../../tests/functions.sh

##### Deleting Previous Results 

echo "  + Deleting previous results..."
rm -rf _korali_result*; check_result

##### Running Test

# Not running this example until libGP is fixed
#python3 ./run-surrogates.py; check_result

  
  
