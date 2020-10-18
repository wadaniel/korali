#!/bin/bash

# Checking necessary env variables
if [ -z $F77 ]; then
  echo "[Error] The environment \$F77 must be defined and point to a Fortran 77 compiler."
  exit -1
fi

if [ -z $CC ]; then
  echo "[Error] The environment \$CC must be defined and point to a C compiler."
  exit -1
fi

# Setting NEK-specific env variables
export NEK5000_DIR=${PWD}/_deps/nek5000
export NEK_SOURCE_ROOT=${NEK5000_DIR}

# Getting code
rm -rf _deps
git clone --depth 1 --recursive https://github.com/Nek5000/Nek5000.git ${NEK5000_DIR}

# Building tools
pushd ${NEK5000_DIR}/tools
./maketools genmap; ./maketools genbox
popd

# Building example
cp _config/* ${NEK5000_DIR}/examples/turbChannel
pushd ${NEK5000_DIR}/examples/turbChannel
MPI=0 FFLAGS='-O3 -g' CFLAGS='-O3 -g' ${NEK5000_DIR}/bin/nekconfig -build-dep;
MPI=0 FFLAGS='-O3 -g' CFLAGS='-O3 -g' ${NEK5000_DIR}/bin/nekconfig;
make -j 6 lib usrfile 
popd

# Copying work files into the work directory
rm -rf _work
mkdir _work
cp ${NEK5000_DIR}/examples/turbChannel/* _work
  

