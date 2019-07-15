#!/bin/bash

##############################################################################
# Brief: Plot all results from tutorials (see REG-002) for basic sanity check. 
# Type: Regression Test 
# Description:
# This test plots all results in the /tutorials/python/ folder to make sure
# the typical use cases work.
###############################################################################

source ../functions.sh

cd $curdir/../../tutorials/python

logEcho "[Korali] Beginning plotting tests"                                   
                                                                                
for dir in ./_*                                                                 
do                                                                              
  logEcho "-------------------------------------"
  logEcho " Plotting results from $dir ..."
  logEcho "-------------------------------------"
  python3 -m korali.plotter --test --dir "${dir}" >> $logFile
  check_result
                     
  python3 -m korali.plotter --test --live --dir "${dir}" >> $logFile
  check_result
                     
  #python3 -m korali.plotter --test --evolution --dir "${dir}" >> $logFile
  #check_result
done 
