#!/bin/bash

rm -rf _deps

# clone CUP
git clone -b amr2 --recursive git@gitlab.ethz.ch:mavt-cse/CubismUP_2D.git _deps/CUP-2D

make -C _deps/CUP-2D/makefiles -j

