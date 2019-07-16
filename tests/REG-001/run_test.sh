#!/bin/bash

##############################################################################
# Brief: Check that solvers initialize with correct default values.
# Type: Regression Test 
# Description:
# This test instantiates all solvers in a secondary python script and 
# valdiates the default values.
###############################################################################

###### Auxiliar Functions and Variables #########

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

