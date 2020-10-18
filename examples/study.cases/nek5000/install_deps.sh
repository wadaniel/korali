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

# Getting source code
rm -rf _deps
git clone --depth 1 --recursive https://github.com/Nek5000/Nek5000.git ${NEK5000_DIR}

# Modifying nek5000 as to not exit on success
pushd ${NEK5000_DIR}/core
cat drive1.f | sed -e 's/call exitt0()/! call exitt0()/g' > drive1.f.tmp
mv drive1.f.tmp drive1.f
popd

# Building turbChannel
cp _config/* ${NEK5000_DIR}/examples/turbChannel
cp _model/turbChannel.usr ${NEK5000_DIR}/examples/turbChannel
pushd ${NEK5000_DIR}/examples/turbChannel
MPI=0 FFLAGS='-O3 -g' CFLAGS='-O3 -g' ${NEK5000_DIR}/bin/nekconfig -build-dep;
MPI=0 FFLAGS='-O3 -g' CFLAGS='-O3 -g' ${NEK5000_DIR}/bin/nekconfig;
make -j 6 lib usrfile 
popd
  

