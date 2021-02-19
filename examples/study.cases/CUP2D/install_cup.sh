#!/bin/bash

rm -rf _deps

# clone AMR version
git clone -b amr2 --recursive git@gitlab.ethz.ch:mavt-cse/CubismUP_2D.git _deps/CUP-2D

# clone non-amr version
# git clone --recursive git@gitlab.ethz.ch:mavt-cse/CubismUP_2D.git _deps/CUP-2D

make -C _deps/CUP-2D/makefiles -j

cp _deps/CUP-2D/makefiles/cup.cflags.txt _deps/CUP-2D/makefiles/cup.libs.txt _deps/CUP-2D/makefiles/libcup.a ./ 

