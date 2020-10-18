#!/bin/bash

if [ -z $F77 ]; then
  echo "[Error] The environment \$F77 must be defined and point to a Fortran 77 compiler."
  exit -1
fi

if [ -z $CC ]; then
  echo "[Error] The environment \$CC must be defined and point to a C compiler."
  exit -1
fi

export NEK5000_DIR=${PWD}/_deps/nek5000
export NEK_SOURCE_ROOT=${NEK5000_DIR}

rm -rf _deps
git clone --depth 1 --recursive https://github.com/Nek5000/Nek5000.git ${NEK5000_DIR}
cd ${NEK5000_DIR}/tools; ./maketools genmap; ./maketools genbox
cd ${NEK5000_DIR}/examples/turbChannel; MPI=0 FFLAGS='-O3 -Ofast -g' CFLAGS='-Ofast -g' ${NEK5000_DIR}/bin/makenek turbChannel -build-dep; make lib usrfile


