#!/usr/bin/env bash

# If clang-format is not installed, run the installation script
clangFormatBin=$PWD/../../prereqs/clang-format
if [ ! -f $clangFormatBin ]; then
 pushd ../..
 ./prereqs/install_clang.sh
 popd
fi

# If run-clang-format is not installed, clone it
if [ ! -f  run-clang-format/run-clang-format.py ]; then

 git clone https://github.com/Sarcasm/run-clang-format.git
 if [ ! $? -eq 0 ]; then
  echo "[Korali] Error installing run-clang-format."
  exit 1
 fi

fi

clangFormatCmd=../../prereqs/clang-format
if [ ! -f $clangFormatCmd ]; then
 clangFormatCmd=clang-format
fi

runClangCmd=run-clang-format/run-clang-format.py

python3 $runClangCmd --clang-format-executable $clangFormatCmd -r ../../source --extensions _cpp,_hpp > /dev/null

if [ ! $? -eq 0 ]; then
 echo "[Korali] C++ Code formatting is incorrect. Please run tools/dev-tools/correct_style.sh to fix it."
 exit -1
else
 echo "[Korali] C++ Code formatting is correct."
 exit 0
fi

