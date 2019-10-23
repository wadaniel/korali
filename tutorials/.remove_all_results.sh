#!/bin/bash

for dir in */
do
  echo "[Korali] Delete results in folder ${dir} ..."
  pushd $dir &> /dev/null
  sh .remove_results.sh
  if [ ! $? -eq 0 ]; then exit -1; fi
  popd &> /dev/null
done
