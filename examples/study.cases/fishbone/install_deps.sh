#!/bin/bash

baseDir=$PWD

rm -rf _deps
git clone git@github.com:cselab/aphros-dev.git _deps/aphros_src --recursive

pushd _deps/aphros_src/deploy
./install_setenv --profile daint ${baseDir}/_deps/aphros
. ap.setenv
mkdir build
cd build
cmake ..
make -j6
make install
popd

pushd _deps/aphros_src/src
. ap.setenv
make -j6
popd
