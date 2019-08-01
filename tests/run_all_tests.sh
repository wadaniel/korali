#!/bin/bash

source functions.sh

#Running tutorials

for dir in */
do
  if [ ! -f "$dir/run_test.sh" ]; then continue; fi
  echo "[Korali] Running test in folder ${dir} ..." 
  pushd $dir
  ./run_test.sh
  check_result
  popd 
done

echo "[Korali] All tests ran correctly."
