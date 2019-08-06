#!/bin/bash

source ../functions.sh

pushd ../../tutorials/

logEcho "[Korali] Beginning plotting tests"                                   

for dir in ./a*/_result_*/
do                                                                              
  logEcho "-------------------------------------"
  logEcho " Plotting results from $dir ..."
  logEcho "-------------------------------------"
  python3 -m korali.plotter --test --dir "${dir}" >> $logFile 2>&1
  check_result
                     
  python3 -m korali.plotter --test --all --gen 1000 --dir "${dir}" >> $logFile 2>&1
  check_result
  
done 

popd
