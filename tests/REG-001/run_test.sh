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


pushd ../../tutorials

logEcho "[Korali] Beginning tutorial tests..."

for dir in ./*/
do
  logEcho "-------------------------------------"
  logEcho " Entering Folder: $dir"
  pushd $dir >> $logFile 2>&1
  
  log "[Korali] Removing any old result files..."
  rm -rf _korali_results >> $logFile 2>&1
  check_result
  
  for file in *.py
  do
    if [ ! -f $file ]; then continue; fi

    logEcho "  + Running File: ${file%.*}"

    log "[Korali] Adding Random Seed..."
    
    resultPath="_result_${file%.*}"
    cat $file | sed -e 's/k.run()/k\[\"General\"\]\[\"Random Seed\"\] = 0xC0FFEE; k.run()/g' \
                    -e 's/k.run()/k\[\"General\"\][\"Results Output\"\][\"Path\"\] = \"'$resultPath'\"; k.run()/g' > tmp.py
    check_result
    
    python3 ./tmp.py >> $logFile 2>&1
    check_result

    rm ./tmp.py >> $logFile 2>&1
    check_result
  done
  
  popd >> $logFile 2>&1
  logEcho "-------------------------------------"
done

popd

