#!/bin/bash

function check_result()
{
 if [ ! $? -eq 0 ]
 then
  echo "[Korali] Error running test."
  exit -1
 fi 
}

#Running tutorials

for dir in */
do
  echo "[Korali] Running test in folder ${dir} ..." 
  pushd $dir
  ./run_test.sh
  check_result
  popd 
done

echo "[Korali] All tests ran correctly."
