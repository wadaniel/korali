#!/bin/bash

##############################################################################
# Brief: Compare python and cxx runs to ensure they produce same results.
# Type: Regression Test 
# Description:
# This test compares the results (json files) produced in /tutorials/python/ 
# with the results from /tutorials/cxx/
###############################################################################

###### Auxiliar Functions and Variables #########

source ../functions.sh

############# STEP 1 ##############

logEcho "[Korali] Beginning comparison tests"

for file in *.py
do
  logEcho "-------------------------------------"
  logEcho " Running $file"
  logEcho "-------------------------------------"
  ./"$file" >> $logFile
  check_result
done
