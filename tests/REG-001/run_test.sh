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

###### Auxiliar Functions #########

function check_result()
{
 if [ ! $? -eq 0 ]
 then
  echo "[Korali] Error running test. Please check $logfile."
  exit -1
 fi 
}

############# STEP 1 ##############

curdir=$PWD
logfile=$curdir/test.log

cd $curdir/../../tutorials/python

echo "[Korali] Beginning python tests" > $logfile

for file in *.py
do
  ./"$file" >> $logfile 2>&1
  check_result
done

############# STEP 2 ##############

cd $curdir/../../tutorials/cxx

make clean >> $logfile 2>&1
check_result

make -j 4 >> $logfile 2>&1
check_result

for file in *.cpp
do
  ./"${file%.*}" >> $logfile 2>&1
  check_result
done

############# STEP 3 ##############

cd $curdir/../../tutorials/python

echo "[Korali] Beginning plotting tests" > $logfile

logfile=$PWD/test.log                                                           
                                                                                
for dir in ./_*                                                                 
do                                                                              
    python3 -m korali.plotter --test --dir "${dir}" >> $logfile 2>&1                   
    #python3 -m korali.plotter --evolution --dir "${dir}" >> $logfile 2>&1
    check_result
done 
