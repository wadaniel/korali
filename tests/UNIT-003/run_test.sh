#!/bin/bash

source ../functions.sh

echo "[Korali] Running Partial Run Tests..."

pushd ../../tutorials/advanced/2.partial.runs
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

  echo "  + Running File: $file"
  ./$file
  check_result
done

echo "-------------------------------------"

popd

