#!/usr/bin/env bash

clangFormatCmd=llvm/bin/clang-format
if [ ! -f $clangFormatCmd ]; then
 clangFormatCmd=clang-format
fi

runClangCmd=run-clang-format/run-clang-format.py

command="python3 $runClangCmd --clang-format-executable $clangFormatCmd -r ../../source --extensions _cpp,_hpp"

$command
