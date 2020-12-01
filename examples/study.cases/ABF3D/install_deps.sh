#!/bin/bash

rm -rf _deps
git clone git@github.com:cselab/msode.git _deps/msode --recursive
mkdir _deps/msode/build
pushd _deps/msode/build
cmake .. -DUSE_SMARTIES=OFF
make -j6
popd

