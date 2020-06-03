#!/bin/bash

source functions.sh

echo "" > $logFile

#Running tutorials

for dir in */
do
  echo "[Korali] Running test in folder ${dir} ..."
  pushd $dir
  ./run_test.sh 2>&1 | tee $logFile
  if [ ! $? -eq 0 ]; then exit -1; fi
  popd
done

echo "[Korali] All tests ran correctly."
