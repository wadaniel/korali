#!/bin/bash

source ../functions.sh

logEcho "[Korali] Beginning Stat Test 002"

for file in run*.py
do
  logEcho "-------------------------------------"
  logEcho " Running $file"
  logEcho "-------------------------------------"
  ./"$file" 
  check_result
done

log "[Korali] Removing results..."
rm -rf _result_run-* 
check_result
