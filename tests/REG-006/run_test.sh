#!/bin/bash

source ../functions.sh

pushd ../../tutorials/

logEcho "[Korali] Beginning profiling tests..."

for dir in ./a*/
do
  if [ -f "${dir}/profiling.json" ]; then
   logEcho "----------------------------------------------"
   logEcho " Processing profiler information from $dir ..."
   logEcho "----------------------------------------------"
   python3 -m korali.profiler --test --dir "${dir}" 
   check_result
  fi
done

popd
