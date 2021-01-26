#!/usr/bin/env bash

###### Check if necessary python modules are installed ######
python3 -m pip show mujoco_py
if [ $? -ne 0 ]; then
 echo "[Korali] mujoco_py not found, aborting test"
 exit 0
fi



##### Running Test

pushd _inverted_double_pendulum > /dev/null
exit_code=$?

./.run_test.sh
exit_code=$(( $exit_code || $? ))

popd > /dev/null
exit_code=$(( $exit_code || $? ))


exit $exit_code