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
# 3 - Operation: plot all results produced in Step 1. 
#     Expected result: all plotting scripts will run for less than 30 secs and
#     rc = 0.
###############################################################################

###### Auxiliar Functions and Variables #########

curdir=$PWD
logFile=$curdir/test.log
echo "" > $logFile

function check_result()
{
 if [ ! $? -eq 0 ]
 then
  echo "[Korali] Error running test. Please check $logFile."
  exit -1
 fi 
}

# Logging and printing function.
function logEcho ()
{
 echo "$1"
 echo "$1" >> $logFile
}

############# STEP 1 ##############

logFile "[Korali] Beginning python tests"

for file in *.py
do
  logEcho "-------------------------------------"
  logEcho " Running $file"
  logEcho "-------------------------------------"
  ./"$file" >> $logFile 2>&1
  check_result
done

############# STEP 2 ##############

#make clean >> $logFile 2>&1
#check_result

#make -j 4 >> $logFile 2>&1
#check_result

#for file in *.cpp
#do
#  logEcho "-------------------------------------"
#  logEcho " Running $file"
#  logEcho "-------------------------------------"
#  ./"${file%.*}" >> $logFile 2>&1
#  check_result
#done
