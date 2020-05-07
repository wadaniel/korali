#!/bin/bash

###### Auxiliar Functions and Variables #########

source ../../../tests/functions.sh

##### Getting Test Type

getTestMode $1

##### Deleting Previous Results

logEcho "  + Deleting previous results..."
rm -rf _korali_result*
check_result

##### Running Tests

for file in run-*.py
do

 if [ ${file: -3} == ".py" ]; then
 
   logEcho "  + Processing File: $file"
 
   testFile=__test${file}
   logEcho "  + Creating Test File: $testFile"
 
   cp $file $testFile
   check_result
   
   if [[ $testMode == "fast" ]]; then
     logEcho "  + Modifying test file for fast execution: $testFile"
     sed  -e 's%\["Burn In"\] =%["Burn In"] = 5 #%g' \
          -e 's%\["Max Samples"\] =%["Max Samples"] = 20 #%g' $file > $testFile 
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
