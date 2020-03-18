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
git submodule update --init --recursive
popd

# Building User Manual
pushd manual
check

pushd builder
check

python3 ./buildTutorials.py 
python3 ./buildTests.py 
python3 ./buildModules.py 
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

# Do Doxygen only if Linux system
if [[ "$OSTYPE" == "linux-gnu" ]]; then

# Installing Doxygen
if ! test -f "doxygen"; then
  wget http://doxygen.nl/files/doxygen-1.8.17.linux.bin.tar.gz
  check

  tar -xzvf doxygen-1.8.17.linux.bin.tar.gz
  check

  mv doxygen-1.8.17/bin/doxygen .
  check

  rm -r doxygen-1.8.17/
  check

  rm doxygen-1.8.17.linux.bin.tar.gz
  check
fi

# Running Doxygen
./doxygen doxygen.config
check

fi

echo "[Korali] Webpage Build complete."
