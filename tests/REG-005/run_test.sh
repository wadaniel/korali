#!/bin/bash

source ../functions.sh

#################################################
# Execute Solver Scripts
#################################################

logEcho "[Korali] Beginning termination criteria tests"


criteria=(
"Max Generations" 
"Max Generations" 
"Max Model Evaluations" 
"Max Infeasible Resamplings" 
"Max Value" 
"Min Value Difference Threshold"
"Min Standard Deviation" 
"Max Standard Deviation"
"Min Standard Deviation Step Factor"
"Max Condition Covariance Matrix" 
"Min Value")

values=(
0     # Max Generations
2     # Max Generations
0     # Max Model Evaluations
1     # Max Infeasible Resamplings
-0.2  # Max Value
0.1   # Min Value Difference Threshold
0.1   # Min Standard Deviation
0.9   # Max Standard Deviation
1.5   # Max Condition Covariance
-1.0  # Min Value
-1.0  # Min Standard Deviation Step Factor
)

for ((i=0;i<${#criteria[@]};++i)); do

  logEcho "-------------------------------------"
  logEcho "Running File: cmaes_termination.py"

  python3 ./cmaes_termination.py --criteria "${criteria[$i]}" --value ${values[$i]} >> $logFile 2>&1
  check_result

  log "[Korali] Removing results..."
  rm -rf "_korali_result" >> $logFile 2>&1
  check_result

  logEcho "-------------------------------------"

done
