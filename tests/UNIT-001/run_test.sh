#!/bin/bash

source ../functions.sh

archString=`uname -a`
if [[ $archString == *"Darwin"* ]]; then
  echo "Skipping C++ tests on Darwin"
  exit 0
fi

logEcho "[Korali] Running C++ Test..."

pushd ../../tutorials/c2-running-cxx/
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

