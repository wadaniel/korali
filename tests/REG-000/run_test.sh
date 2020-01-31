#!/bin/bash

###### Auxiliar Functions and Variables #########

source ../functions.sh

############# STEP 1 ##############

logEcho "[Korali] Checking Pip Installation"
python3 -m pip check korali
check_result

############# STEP 2 ##############

logEcho "[Korali] Checking korali.plotter"
python3 -m korali.plotter --check
check_result

logEcho "[Korali] Checking korali.cxx"
python3 -m korali.cxx --cflags
check_result

python3 -m korali.cxx --compiler
check_result

python3 -m korali.cxx --libs
check_result
