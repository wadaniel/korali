#!/bin/bash

pushd ../../tools/style

./style_cxx.sh check
if [ $? -ne 0 ]; then
 echo "[Korali] Error: Could not validate C++ style. clang-format version used:" 
 clang-format --version
 exit 1
fi
