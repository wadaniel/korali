#!/bin/bash

##############################################################################
# Brief: Running Korali with C++
# Type: Unit Test 
# Description:
# Run a C++ model using a C++ Korali Application
###############################################################################

###### Auxiliar Functions and Variables #########

source ../functions.sh

############# STEP 1 ##############

logEcho "[Korali] Running C++ Test..."

pushd ../../tutorials/b3-running-cxx/
dir=$PWD

logEcho "-------------------------------------"
logEcho " Entering Folder: $dir"

log "[Korali] Removing any old result files..."
rm -rf _korali_results >> $logFile 2>&1
check_result

log "[Korali] Compiling test case..."
make clean >> $logFile 2>&1
check_result

make -j >> $logFile 2>&1
check_result

for file in *.cpp
do
  if [ ! -f $file ]; then continue; fi

  execName=${file%.*}
  logEcho "  + Running File: $execName"
  ./$execName >> $logFile 2>&1
  check_result
done

logEcho "-------------------------------------"

popd

