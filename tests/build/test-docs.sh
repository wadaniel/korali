#!/bin/bash

pushd ../../docs

make html 2>&1 | tee /dev/stderr | cat > build.log
if [ $? -ne 0 ]; then
 echo "[Korali] Error: Could not generate documentation. Check docs/build.log" 
 exit 1
fi

cat build.log | grep -i "warning"
if [ $? -eq 0 ]; then
 echo "[Korali] Error: Warning detected in documentation generation" 
 exit 1
fi

cat build.log | grep -i "error"
if [ $? -eq 0 ]; then
 echo "[Korali] Error: error detected in documentation generation" 
 exit 1
fi


