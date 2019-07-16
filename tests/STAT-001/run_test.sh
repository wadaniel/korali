#!/bin/bash

##############################################################################
# Brief: Check Sampling Statistics
# Type: Regression Test 
# Description:
# This test checks the sampling statistics of MCMC and TMCMC of the results
# taken from the tutorial folder (example 2a, direct evaluation)
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

