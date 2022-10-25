#!/bin/bash

###### Check if necessary python modules are installed ######

python3 -m pip install gym
exit_code=$?

exit_code=$(( $exit_code || $? ))

##### Deleting Previous Results 

echo "  + Deleting previous results..."
rm -rf _result*
exit_code=$(( $exit_code || $? ))

##### Running Test

echo "Running ${file} ..."
python3 run-vracer.py --env HalfCheetah-v4 --exp 1000
exit_code=$(( $exit_code || $? ))
