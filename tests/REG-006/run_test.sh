#!/bin/bash

source ../functions.sh

pushd ../../tutorials/

echo "[Korali] Beginning profiling tests..."

for dir in ./a*/
do
  if [ -f "${dir}/profiling.json" ]; then
   echo "----------------------------------------------"
   echo " Processing profiler information from $dir ..."
   echo "----------------------------------------------"
   python3 -m korali.profiler --test --dir "${dir}" 
   check_result
  fi
done

popd
