#!/bin/bash

source functions.sh

#Running tutorials

dir=STAT-005
echo "[Korali] Running test in folder ${dir} ..."
pushd $dir
./run_test.sh
if [ ! $? -eq 0 ]; then exit -1; fi
popd

echo "[Korali] All tests ran correctly."

# for dir in */
# do
#   echo "[Korali] Running test in folder ${dir} ..."
#   pushd $dir
#   ./run_test.sh
#   if [ ! $? -eq 0 ]; then exit -1; fi
#   popd
# done
#
# echo "[Korali] All tests ran correctly."
