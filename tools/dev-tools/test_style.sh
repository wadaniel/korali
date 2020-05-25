#!/usr/bin/env bash

clangFormatCmd=llvm/bin/clang-format
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

