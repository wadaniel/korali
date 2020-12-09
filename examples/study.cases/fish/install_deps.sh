#!/bin/bash

if [ -z $HYPRE_ROOT ]; then
  echo "[Error] The environment \$HYPRE_ROOT must point to the installation path for Hypre"
  exit -1
fi

if [ -z $HDF5_ROOT ]; then
  echo "[Error] The environment \$HDF5_ROOT must point to the installation path for HDF5"
  exit -1
fi

if [ -z $GSL_ROOT_DIR ]; then
  echo "[Error] The environment \$GSL_ROOT_DIR must point to the installation path for GSL"
  exit -1
fi

#if [ -z $FFTW_ROOT ]; then
#  echo "[Error] The environment \$FFTW_ROOT must point to the installation path for FFTW"
#  exit -1
#fi

if [ -z $CUBISM_BLOCK_SIZE ]; then
  echo "[Warning] The environment \$CUBISM_BLOCK_SIZE not defined. Using default: 8"
  CUBISM_BLOCK_SIZE=8
fi
CUBISM_NTHREADS=0

rm -rf _deps

git clone --branch AMRKorali --recursive git@gitlab.ethz.ch:mavt-cse/CubismUP_2D.git _deps/cubism
pushd _deps/cubism/makefiles
cat Makefile | sed -e 's/bs ?= /bs ?= '$CUBISM_BLOCK_SIZE'#/g' \
                   -e 's/nthreads ?=/nthreads ?= '$CUBISM_NTHREADS'#/g' > Makefile.new
mv Makefile.new Makefile
make -j6 gpu=false
cp libcubism.a cubism.cflags.txt cubism.libs.txt ../../..
popd

