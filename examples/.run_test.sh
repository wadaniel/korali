#!/bin/bash

###### Auxiliar Functions and Variables #########

source ../tests/functions.sh

exampleDirs=`find -type d -not -path "*/_*" | sort | awk '$0 !~ last "/" {print last} {last=$0} END {print last}'`

for dir in $exampleDirs
do
  echo " + Entering Folder: $dir"
  pushd $dir; check_result
  
  ./.run_test.sh; check_result
  
  popd; check_result
done

exit 0