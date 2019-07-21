#!/bin/bash

##############################################################################
# Brief: Re-run all Python tutorials for basic sanity check. 
# Type: Regression Test 
# Description:
# This test finds and runs tutorials in the /tutorials folder to make sure
# the typical use cases still work.
# Steps: 
# 1 - Operation: List and run all .py scripts in the tutorials/python folder.
#     Expected result: all of the .py scripts will run for less than 20 secs
#     and rc = 0.  
###############################################################################

###### Auxiliar Functions and Variables #########

source ../functions.sh

############# STEP 1 ##############

logEcho "[Korali] Copying Tutorials..."
rm -rf ./tutorials >> $logFile 2>&1
check_result

cp -R ../../tutorials . >> $logFile 2>&1
check_result

pushd tutorials

logEcho "[Korali] Beginning python tests"

for dir in ./*                                                                 
do
  logEcho "-------------------------------------"
  logEcho " Entering Tutorial: $dir"
  pushd $dir/python >> $logFile 2>&1
  
  log "[Korali] Removing any old result files..."
  rm -rf _korali_results >> $logFile 2>&1
  check_result
  
  for file in *.py
  do
    logEcho "  + Running File: ${file%.*}"

    log "[Korali] Adding Random Seed..."
    cat $file | sed -e 's/k.run()/k\[\"General\"\]\[\"Random Seed\"\] = 0xC0FFEE; k.run()/g' > tmp
    check_result
    
    log "[Korali] Replacing File..."
    mv tmp $file
    check_result

    log "[Korali] Running $file..."
    python3 ./$file >> $logFile 2>&1
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

