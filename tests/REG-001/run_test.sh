#!/bin/bash

function check_result()
{
 if [ ! $? -eq 0 ]
 then
  echo "[Korali] Error running test. Please check $logfile."
  exit -1
 fi 
}

#Running tutorials

curdir=$PWD
logfile=$curdir/test.log

cd $curdir/../../tutorials/python

echo "[Korali] Beginning python tests" > $logfile

for file in *.py
do
  ./"$file" >> $logfile 2>&1
  check_result
done

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
