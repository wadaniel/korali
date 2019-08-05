#!/bin/bash

##############################################################################
# Brief: Test correct plotting for all results from tutorial runs.
# Type: Regression Test 
# Description:
# This test plots all results in the /tutorials/python/ folder to make sure
# the typical use cases work. Note: flag --all untested.
###############################################################################

source ../functions.sh

pushd ../../tutorials/

logEcho "[Korali] Beginning plotting tests"                                   

for dir in ./*/_result_*/
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
