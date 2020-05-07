#!/bin/bash

function check()
{
 if [ ! $? -eq 0 ]
 then
  echo "[Korali] Error building site."
  exit -1
 fi 
}

# Building Korali source
pushd ../source
check

python3 ./build.py
check

popd
check

# Installing shpinx, mkdocs, and materials theme
python3 -m pip install sphinx --user
check

python3 -m pip install sphinx_rtd_theme --user
check

python3 -m pip install Pygments --user
check

# Getting korali-apps submodule
pushd ..
git submodule update --init --recursive --remote
popd

# Building User Manual
pushd manual
check

pushd builder
check

python3 ./buildTutorials.py 
check

python3 ./buildTests.py 
check

python3 ./buildModules.py 
check

python3 ./buildTools.py
check

popd
check

make html
check

popd

# Inserting user manual into website

mkdir -p web/docs
check 

cp -r manual/.build/html/* web/docs
check

# Running Doxygen
doxygen doxygen.config
check

#fi

echo "[Korali] Webpage Build complete."
