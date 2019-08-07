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
rm -rf ../tutorials >> $logFile 2>&1
check_result

cp -R ../../tutorials ../ >> $logFile 2>&1
check_result

pushd ../tutorials

log "[Korali] Removing guides..."
rm -r g*
check_result

log "[Korali] Removing other files..."
rm README.md
check_result

logEcho "[Korali] Beginning python tests"

for dir in ./*                                                                 
do
  logEcho "-------------------------------------"
  logEcho " Entering Tutorial: $dir"

  if [ ! -d "$dir/python" ]; then
    echo "  + No folder named 'python' found inside $dir"
    continue
  fi

  pushd $dir/python >> $logFile 2>&1
  

  log "[Korali] Removing any old result files..."
  rm -rf _korali_results >> $logFile 2>&1
  check_result
  
  for file in run-*.py
  do
    
    # In case there are no files matching the pattern 'run-*.py', break
    if [ ! -f "$file" ]; then
      echo "  + No run*-.py file found!"
      continue
    fi
    

    logEcho "  + Running File: ${file%.*}"

    log "[Korali] Adding Random Seed..."
    
    resultPath="_result_${file%.*}"
    cat $file | sed -e 's/k.run()/k\[\"General\"\]\[\"Random Seed\"\] = 0xC0FFEE; k.run()/g' \
                    -e 's/k.run()/k\[\"General\"\][\"Results Output\"\][\"Path\"\] = \"'$resultPath'\"; k.run()/g' > tmp
    check_result
    
    log "[Korali] Replacing File..."
    mv tmp $file
    check_result

    log "[Korali] Moving Results..."
    rm -rf $resultPath >> $logFile 2>&1
    check_result

    log "[Korali] Running $file..."
    python3 ./$file >> $logFile 2>&1
    check_result
  done
  
  popd >> $logFile 2>&1
  logEcho "-------------------------------------"
done

popd

