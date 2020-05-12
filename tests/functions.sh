###### Auxiliar Functions and Variables #########

curdir=$PWD
logFile=$curdir/test.log
echo "" > $logFile

# Function to check whether the last command was correctly executed
function check_result()
{
 if [ ! $? -eq 0 ]
 then
  echo "[Korali] Error running test: (check log: $logFile)"
  tail $logFile
  exit -1
 fi 
}

# Logging and printing function.
function logEcho ()
{
 echo "$1"
 echo "$1" >> $logFile
}

# Logging function.
function log ()
{
 echo "$1" >> $logFile
}


# Function to get test type
function getTestMode()
{
 testMode=undefined

 if [[ "$1" == "--fast" ]]; then testMode="fast"; fi
 if [[ "$1" == "--full" ]]; then testMode="full"; fi

 if [[ $testMode == "undefined" ]]; then
  echo "Error: must define whether the test is meant to run in --fast or --full mode."
  exit -1
 fi
}