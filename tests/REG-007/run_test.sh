#!/bin/bash

source ../functions.sh


#################################################
# MIN/MAX Optimizer
#################################################

logEcho "[Korali] Beginning minimizing mazimizing tests"

logEcho "-------------------------------------"
logEcho "Testing Maximizing"
logEcho "Running File: run-maxcmaes1.py"

python3 ./run-maxcmaes1.py >> $logFile 2>&1
check_result

log "[Korali] Removing results..."
rm -rf "_korali_result" >> $logFile 2>&1
check_result

logEcho "-------------------------------------"

logEcho "-------------------------------------"
logEcho "Testing Maximizing"
logEcho "Running File: run-maxcmaes2.py"

python3 ./run-maxcmaes2.py >> $logFile 2>&1
check_result

log "[Korali] Removing results..."
rm -rf "_korali_result" >> $logFile 2>&1
check_result

logEcho "-------------------------------------"

logEcho "-------------------------------------"
logEcho "Testing Minimizing"
logEcho "Running File: run-mincmaes1.py"

python3 ./run-mincmaes1.py >> $logFile 2>&1
check_result

log "[Korali] Removing results..."
rm -rf "_korali_result" >> $logFile 2>&1
check_result

logEcho "-------------------------------------"

logEcho "-------------------------------------"
logEcho "Testing Minimizing"
logEcho "Running File: run-mincmaes2.py"

python3 ./run-mincmaes2.py >> $logFile 2>&1
check_result

log "[Korali] Removing results..."
rm -rf "_korali_result" >> $logFile 2>&1
check_result

logEcho "-------------------------------------"

logEcho "-------------------------------------"
logEcho "Testing Minimizing C++"
logEcho "Build & Running File: run-mincmaes1.cpp"

make
python3 ./run-mincmaes2.py >> $logFile 2>&1
check_result

log "[Korali] Removing results..."
rm -rf "_korali_result" >> $logFile 2>&1
check_result

make clean
logEcho "-------------------------------------"
