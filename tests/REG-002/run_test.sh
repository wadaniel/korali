#!/bin/bash

##############################################################################
# Brief: Re-run all C++ tutorials for basic sanity check. 
# Type: Regression Test 
# Description:
# This test finds and runs tutorials in the /tutorials folder to make sure
# the typical use cases still work.
# Steps: 
# 1 - Operation: List, compile, and run all .cpp files in the tutorials/cxx
#     folder. 
#     Expected result: all of the .py scripts will run for less than 20
#     secs and rc = 0.
###############################################################################

###### Auxiliar Functions and Variables #########

source ../functions.sh

############# STEP 1 ##############

logEcho "[Korali] Copying Tutorials..."
rm -rf ./tutorials >> $logFile 2>&1
check_result

cp ../../tutorials . -r >> $logFile 2>&1
check_result

pushd tutorials

logEcho "[Korali] Beginning C++ tests"

for dir in ./*                                                                 
do
  logEcho "-------------------------------------"
  logEcho " Entering Tutorial: $dir"

  pushd $dir/cxx >> $logFile 2>&1

  logEcho "  + Compiling Tutorial..."
  
  make clean >> $logFile 2>&1
  check_result
  
  make -j >> $logFile 2>&1
  check_result

  log "[Korali] Removing any old result files..."
  rm -rf _korali_results >> $logFile 2>&1
  check_result
  
  for file in *.cpp
  do
    logEcho "  + Running File: $file..."
    
    log "[Korali] Running $file..."
    ./"${file%.*}" >> $logFile 2>&1
    check_result
    
    log "[Korali] Moving Results..."
    rm -rf "_result_${file%.*}" >> $logFile 2>&1
    check_result
    
    mv _korali_result "_result_${file%.*}" >> $logFile 2>&1
    check_result
  done
  
  popd >> $logFile 2>&1
  logEcho "-------------------------------------"
done

popd

