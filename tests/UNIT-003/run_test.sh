#!/bin/bash

##############################################################################
# Brief: Check Chain Mean and Chain Covariance in MCMC
# Type: Unit Test
# Description:
# This test calculates the mean and standard deviation of the samples produced
# and compares the values with the internal variables of MCMC. For this the
# results from the tutorials are reused.
################################################################################

###### Auxiliar Functions and Variables #########

source ../functions.sh

############# STEP 1 ##############

logEcho "[Korali] Beginning Unit Test 003"

for file in *.py
do
  logEcho "-------------------------------------"
  logEcho " Running $file"
  logEcho "-------------------------------------"
  ./"$file" >> $logFile
  check_result
done

