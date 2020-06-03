#!/bin/bash

source ../functions.sh

echo "[Korali] Running Multiple Jobs test..."

pushd ../../tutorials/advanced/7.multiple.experiments
check_result
dir=$PWD

echo "-------------------------------------"
echo " Entering Folder: $dir"

echo "[Korali] Removing any old result files..."
rm -rf _korali_results
check_result

for file in *.py
do
  if [ ! -f $file ]; then continue; fi

  for i in $(seq 1 5)
  do
   echo "  + Running File: $file"
   ./$file
   check_result
  done
done

echo "-------------------------------------"

popd

