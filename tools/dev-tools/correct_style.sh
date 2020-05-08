#!/usr/bin/env bash
# File       : correct_style
# Created    : Mon Jan 20 2020 11:29:59 AM (+0100)
# Author     : Fabian Wermelinger (adjusted to Korali by Sergio Martin)
# Description: Utility to adjust source code according to style conventions
# Copyright 2020 ETH Zurich. All Rights Reserved.

SRC_H='.*\._hpp'
SRC_S='.*\._cpp'

# format in-place
find ../../source \
    -regextype 'egrep' \
    -regex "${SRC_H}" -or \
    -regex "${SRC_S}" | \
    xargs -n6 -P2 echo "$@"   
#xargs -n6 -P2 clang-format -style=file -i "$@"
