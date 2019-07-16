#!/bin/bash

##############################################################################
# Brief: Re-run all tutorials for basic sanity check. 
# Type: Regression Test 
# Description:
# This test finds and runs tutorials in the /tutorials folder to make sure
# the typical use cases still work.
# Steps: 
# 1 - Operation: List and run all .py scripts in the tutorials/python folder.
#     Expected result: all of the .py scripts will run for less than 20 secs
#     and rc = 0.  
# 2 - Operation: List, compile, and run all .cpp files in the tutorials/cxx
#     folder. 
#     Expected result: all of the .py scripts will run for less than 20
#     secs and rc = 0.
###############################################################################

###### Auxiliar Functions and Variables #########

source ../functions.sh

############# STEP 1 ##############

cd $curdir/../../tutorials/python

logEcho "[Korali] Removing old result files"
rm -rf _*

logEcho "[Korali] Beginning python tests"

for file in *.py
do
  logEcho "-------------------------------------"
  logEcho " Running $file"
  logEcho "-------------------------------------"
  ./"$file" >> $logFile
  check_result
done

############# STEP 2 ##############

cd $curdir/../../tutorials/cxx

logEcho "[Korali] Removing old result files"
rm -rf _*

logEcho "[Korali] Compiling executables"

make clean >> $logFile
check_result

make -j 4 >> $logFile
check_result

logEcho "[Korali] Beginning c++ tests"

for file in *.cpp
do
  logEcho "-------------------------------------"
  logEcho " Running $file"
  logEcho "-------------------------------------"
  ./"${file%.*}" >> $logFile
  check_result
done
