#!/usr/bin/env bash
# File       : correct_style
# Created    : Mon Jan 20 2020 11:29:59 AM (+0100)
# Author     : Fabian Wermelinger (adjusted to Korali by Sergio Martin and Lucas Amoudruz)
# Description: Utility to adjust source code according to style conventions
# Copyright 2020 ETH Zurich. All Rights Reserved.

fileDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
pushd $fileDir

# If clang-format is not installed, run the installation script
clangFormatBin=$PWD/../../prereqs/clang-format
if [ ! -f $clangFormatBin ]; then
 pushd ../..
 ./prereqs/install_clang.sh
 popd
fi

root="../../"

src_files=`find $root -type f -name "*pp" -not -path "$root/external/*"`

echo $src_files | \
    xargs -n6 -P2 $clangFormatBin -style=file -i "$@"

popd
