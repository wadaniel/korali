#!/bin/bash

###### Auxiliar Functions and Variables #########

source ../../../tests/functions.sh

##### Getting Test Type

testMode=undefined

if [[ "$1" == "--fast" ]]; then testMode="fast"; fi
if [[ "$1" == "--full" ]]; then testMode="full"; fi

if [[ $testMode == "undefined" ]]; then
 echo "Error: must define whether the test is meant to run in --fast or --full mode."
 exit -1
fi


##### Running Tests

for file in run-*.py
  do

    if [ ${file: -3} == ".py" ]; then

      logEcho "  + Processing File: $file"

      testFile=__test${file}
      logEcho "  + Creating Test File: $testFile"

      if [[ $testMode == "fast" ]]; then
         sed  -e 's%granularity = 0.0%granularity = 0.2 #%g' \
              -e 's%granularity = 1%granularity = 20 #%g' \
              $file > $testFile 
         check_result
      fi

      if [[ $testMode == "full" ]]; then
         cp $file $testFile
         check_result
      fi

      logEcho "  + Running Test File: $testFile"
      python3 $testFile >> $logFile 2>&1
      check_result

      logEcho "  + Deleting Test File"
      rm $testFile
      check_result
    fi
    
  done
