#!/bin/bash

source ../functions.sh

############# STEP 1 ##############

logEcho "[Korali] Beginning samping statistics tests..."

for file in run*.py
do
  logEcho "-------------------------------------"
  logEcho " Running $file"
  logEcho "-------------------------------------"
  ./"$file" >> $logFile
  check_result
done

for file in stats*.py
do
  logEcho "-------------------------------------"
  logEcho " Running $file"
  logEcho "-------------------------------------"
  ./"$file" >> $logFile
  check_result
done

