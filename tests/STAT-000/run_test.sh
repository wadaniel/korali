#!/bin/bash

source ../functions.sh

############# STEP 1 ##############

logEcho "[Korali] Beginning defaults tests"

for file in *.py
do
  logEcho "-------------------------------------"
  logEcho " Running $file"
  logEcho "-------------------------------------"
  ./"$file" >> $logFile
  check_result
done

log "[Korali] Removing results..."
rm -rf _korali_result >> $logFile 2>&1
check_result
