#!/usr/bin/env bash
# File       : correct_style
# Created    : Mon Jan 20 2020 11:29:59 AM (+0100)
# Author     : Fabian Wermelinger (adjusted to Korali by Sergio Martin and Lucas Amoudruz)
# Description: Utility to adjust source code according to style conventions
# Copyright 2020 ETH Zurich. All Rights Reserved.

function check()
{
 if [ ! $? -eq 0 ]
 then
  echo "[Korali] Error fixing style."
  exit -1
 fi 
}

fileDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
pushd $fileDir

root="../../"

##############################################
### Correcting C++ Code Style
##############################################

# If clang-format is not installed, run the installation script
clangFormatBin=$PWD/../../prereqs/clang-format
if [ ! -f $clangFormatBin ]; then
 pushd ../..
 ./prereqs/install_clang.sh
 check
 popd
fi

src_files=`find $root -type f -name "*pp" -not -path "$root/external/*"`

echo $src_files | \
    xargs -n6 -P2 $clangFormatBin -style=file -i "$@"

check

##############################################
### Correcting Python Code Style
##############################################

PIP_USER=$(python3 -c "import sys; hasattr(sys, 'real_prefix') or print('--user')")

python3 -m yapf --version
if [ $? -ne 0 ]; then

  echo "[Korali] yapf not found, trying to install it automatically."
  python3 -m pip install $PIP_USER yapf >> $logFile 2>&1; check
fi

src_files=`find $root -type f -name "*.py" -not -path "$root/external/*"`

echo $src_files | \
    xargs -n6 -P2 python3 -m yapf --style=yapf -i "$@"

popd
