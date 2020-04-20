#!/bin/bash

source ../functions.sh

cmaes_criteria=(
"Max Generations" 
"Max Generations" 
"Max Infeasible Resamplings" 
"Min Value Difference Threshold"
"Min Standard Deviation" 
"Max Standard Deviation"
"Max Condition Covariance Matrix" 
)

cmaes_values=(
1     # Max Generations
3     # Max Generations
1     # Max Infeasible Resamplings
0.1   # Min Value Difference Threshold
0.1   # Min Standard Deviation
0.9   # Max Standard Deviation
1.0   # Max Condition Covariance
)

dea_criteria=(
"Max Generations" 
"Max Generations" 
"Max Infeasible Resamplings" 
"Min Value Difference Threshold"
)

dea_values=(
1     # Max Generations
3     # Max Generations
0     # Max Infeasible Resamplings
0.1   # Min Value Difference Threshold
)

tmcmc_criteria=(
"Max Generations" 
"Max Generations" 
"Target Annealing Exponent"
)

tmcmc_values=(
1     # Max Generations
3     # Max Generations
0.6   # Target Annealing Exponent
)

nested_criteria=(
"Max Generations"
"Max Effective Sample Size"
)

nested_values=(
10    # Max Generations
500   # Max Effective Sample Size
)


#################################################
# CMA-ES Termination Criterion Tests
#################################################

logEcho "[Korali] Beginning CMA-ES termination criterion tests"

for ((i=0;i<${#cmaes_criteria[@]};++i)); do

  logEcho "-------------------------------------"
  logEcho "Testing Termination Criterion: ${cmaes_criteria[$i]}"
  logEcho "Running File: cmaes_termination.py"

  python3 ./cmaes_termination.py --criterion "${cmaes_criteria[$i]}" --value ${cmaes_values[$i]} >> $logFile 2>&1
  check_result

  log "[Korali] Removing results..."
  rm -rf "_korali_result" >> $logFile 2>&1
  check_result

  logEcho "-------------------------------------"

done


#################################################
# DEA Termination Criterion Tests
#################################################

logEcho "[Korali] Beginning DEA termination criterion tests"

for ((i=0;i<${#dea_criteria[@]};++i)); do

  logEcho "-------------------------------------"
  logEcho "Testing Termination Criterion: ${dea_criteria[$i]}"
  logEcho "Running File: dea_termination.py"

  python3 ./dea_termination.py --criterion "${dea_criteria[$i]}" --value ${dea_values[$i]} >> $logFile 2>&1
  check_result

  log "[Korali] Removing results..."
  rm -rf "_korali_result" >> $logFile 2>&1
  check_result

  logEcho "-------------------------------------"

done


#################################################
# TMCMC Termination Criterion Tests
#################################################

logEcho "[Korali] Beginning TMCMC termination criterion tests"

for ((i=0;i<${#tmcmc_criteria[@]};++i)); do

  logEcho "-------------------------------------"
  logEcho "Testing Termination Criterion: ${tmcmc_criteria[$i]}"
  logEcho "Running File: tmcmc_termination.py"

  python3 ./tmcmc_termination.py --criterion "${tmcmc_criteria[$i]}" --value ${tmcmc_values[$i]} >> $logFile 2>&1
  check_result

  log "[Korali] Removing results..."
  rm -rf "_korali_result" >> $logFile 2>&1
  check_result

  logEcho "-------------------------------------"

done

#################################################
# Nested Termination Criterion Tests
#################################################

logEcho "[Korali] Beginning Nested termination criterion tests"

for ((i=0;i<${#tmcmc_criteria[@]};++i)); do

  logEcho "-------------------------------------"
  logEcho "Testing Termination Criterion: ${nested_criteria[$i]}"
  logEcho "Running File: nested_termination.py"

  python3 ./nested_termination.py --criterion "${nested_criteria[$i]}" --value ${nested_values[$i]} >> $logFile 2>&1
  check_result

  log "[Korali] Removing results..."
  rm -rf "_korali_result" >> $logFile 2>&1
  check_result

  logEcho "-------------------------------------"

done
