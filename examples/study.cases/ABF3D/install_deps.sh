#!/bin/bash

rm -rf _deps
git clone git@github.com:cselab/msode.git _deps/msode --recursive
mkdir _deps/msode/build
pushd _deps/msode/build
cmake .. -DUSE_SMARTIES=OFF -DLIBBFD_BFD_LIBRARY=$HOME/libs/binutils/lib/libbfd.a
make -j6
popd

rm -rf _config
cp -r _deps/msode/launch_scripts/rl/config _config

