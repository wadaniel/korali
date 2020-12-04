#!/bin/bash

###### Check if necessary python modules are installed ######
python3 -m pip show mujoco_py
if [ $? -ne 0 ]; then
 echo "[Korali] mujoco_py not found, aborting test"
 exit 0
fi

###### Auxiliar Functions and Variables #########

source ../../../tests/functions.sh

##### Running Test

pushd _inverted_double_pendulum; check_result
  ./.run_test.sh; check_result
popd
