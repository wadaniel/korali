#!/bin/bash

source ../functions.sh

pushd ../../tutorials/

logEcho "[Korali] Beginning profiling tests..."

for dir in ./a*/*_result*/
do
  logEcho "----------------------------------------------"
  logEcho " Processing profiler information from $dir ..."
  logEcho "----------------------------------------------"
  python3 -m korali.profiler --test --dir "${dir}" >> $logFile 2>&1
  check_result
done

popd
