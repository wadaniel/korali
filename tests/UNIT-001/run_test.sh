#!/bin/bash

source ../functions.sh

archString=`uname -a`
if [[ $archString == *"Darwin"* ]]; then
  echo "Skipping C++ tests on Darwin"
  exit 0
fi

echo "[Korali] Running C++ Test..."

pushd ../../tutorials/advanced/4.running.cxx/
check_result
dir=$PWD

echo "-------------------------------------"
echo " Entering Folder: $dir"

echo "[Korali] Removing any old result files..."
rm -rf _korali_results
check_result

echo "[Korali] Compiling test case..."
make clean
check_result

make -j
check_result

for file in *.cpp
do
  if [ ! -f $file ]; then continue; fi

  execName=${file%.*}
  echo "  + Running File: $execName"
  ./$execName
  check_result
done

echo "-------------------------------------"

popd

