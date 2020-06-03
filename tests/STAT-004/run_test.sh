#!/bin/bash

source ../functions.sh

logEcho "[Korali] Beginning Stat Test 004"

for file in run-*.py
do
  logEcho "-------------------------------------"
  logEcho " Running $file"
  logEcho "-------------------------------------"
  ./"$file" 
  check_result
done

log "[Korali] Removing results..."
rm -rf _korali_result 
check_result
