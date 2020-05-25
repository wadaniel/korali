#!/usr/bin/env bash
# File       : correct_style
# Created    : Mon Jan 20 2020 11:29:59 AM (+0100)
# Author     : Fabian Wermelinger (adjusted to Korali by Sergio Martin)
# Description: Utility to adjust source code according to style conventions
# Copyright 2020 ETH Zurich. All Rights Reserved.

clangFormat=llvm/bin/clang-format
if [ ! -f $clangFormat ]; then
 clangFormat=clang-format
fi

root="../../source"

src_files=`find $root -type f -name "*pp" -not -path "$root/external/*"`

echo $src_files | \
    xargs -n6 -P2 $clangFormat -style=file -i "$@"
