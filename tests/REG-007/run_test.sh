#!/bin/bash

##############################################################################
# Brief: Re-run all Python example applications for basic sanity check.
# Type: Regression Test
# Description:
# This test finds and runs A Tutorials /tutorials folder to make sure
# the typical use cases still work.
# Steps:
# 1 - Operation: List and run all .py scripts in the tutorials/python folder.
#     Expected result: all of the .py scripts will run for less than 20 secs
#     and rc = 0.
###############################################################################

###### Auxiliar Functions and Variables #########

source ../functions.sh

############# STEP 1 ##############

logEcho "[Korali] Pulling korali-apps repository"
pushd ../../
git submodule update --init --recursive --remote
check_result
popd

logEcho "[Korali] Beginning case study dry run tests..."
pushd ../../tutorials

for dir in ./examples/*/
do
  logEcho "-------------------------------------"
  logEcho " Entering Folder: $dir"
  pushd $dir >> $logFile 2>&1
  check_result

  for file in run-*.{py,sh}
  do
    # In case there are no files matching the pattern 'run-*.py', break
    if [ ! -f "$file" ]; then
      continue
    fi
    
    sed 's/k.run(/k["Dry Run"] = True; k.run(/g' $file > _$file
 
    logEcho "  + Running File: $file"
    if [ ${file: -3} == ".py" ]; then
      python3 _$file >> $logFile 2>&1
      check_result
    fi
    
    if [ ${file: -3} == ".sh" ]; then
      sh _$file >> $logFile 2>&1
      check_result
    fi
    
  rm -f _*.py _*.sh
  check_result

  done

  popd >> $logFile 2>&1
  check_result
  logEcho "-------------------------------------"
done

popd
