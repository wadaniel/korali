#!/usr/bin/env bash

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

root=$fileDir/../..

##############################################
### Testing C++ Code Style
##############################################

# If clang-format is not installed, run the installation script
clangFormatBin=${root}/external/clang-format
if [ ! -f $clangFormatBin ]; then
 pushd ${root}
 ./external/install_clang.sh
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

clangFormatCmd=${root}/external/clang-format
if [ ! -f $clangFormatCmd ]; then
 clangFormatCmd=clang-format
fi

runClangCmd=run-clang-format/run-clang-format.py

python3 $runClangCmd --clang-format-executable $clangFormatCmd -r ${root} --extensions _cpp,_hpp > /dev/null

if [ ! $? -eq 0 ]; then
 echo "[Korali] Error: C++ Code formatting is not normalized."
 echo "[Korali] Solution: Please run $fileDir/correct_style.sh to fix it."
 exit -1
else
 echo "[Korali] C++ Code formatting is correct."
fi

##############################################
### Testing Python Code Style
##############################################

PIP_USER=$(python3 -c "import sys; hasattr(sys, 'real_prefix') or print('--user')")

python3 -m yapf --version > /dev/null
if [ $? -ne 0 ]; then

  echo "[Korali] yapf not found, trying to install it automatically."
  python3 -m pip install $PIP_USER yapf; check
fi

src_files=`find ${root} -type f -not -name "__*" -name "*.py" -not -path "${root}/source/external/*" -not -path "${root}/external/*" -not -path "${root}/tools/dev-tools/*"`

diff=`echo $src_files | xargs -n6 -P2 python3 -m yapf --style=yapf -d "$@"`

if [ ! "$diff" == "" ]; then
 echo "[Korali] Error: Python Code formatting is not normalized."
 echo "[Korali] Solution: Please run $fileDir/correct_style.sh to fix it."
 exit -1
else
 echo "[Korali] Python code formatting is correct."
fi

popd

exit 0
