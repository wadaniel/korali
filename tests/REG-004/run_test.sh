#!/bin/bash

##############################################################################
# Brief: Test correct plotting for all results from tutorial runs.
# Type: Regression Test 
# Description:
# This test plots all results in the /tutorials/python/ folder to make sure
# the typical use cases work.
###############################################################################

source ../functions.sh

pushd ../REG-001/tutorials

logEcho "[Korali] Beginning plotting tests"                                   

for dir in ./*/python/_*                                                                 
do                                                                              
  logEcho "-------------------------------------"
  logEcho " Plotting results from $dir ..."
  logEcho "-------------------------------------"
  python3 -m korali.plotter --test --dir "${dir}" >> $logFile 2>&1
  check_result
                     
  python3 -m korali.plotter --test --live --dir "${dir}" >> $logFile 2>&1
  check_result
                     
done 

popd
