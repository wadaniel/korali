#!/bin/bash

wget 'ftp://ftp.gnu.org/gnu/gsl/gsl-2.6.tar.gz'
tar -xzvf gsl-2.6.tar.gz
echo "MESON_INSTALL=${MESON_INSTALL_PREFIX}"
echo "MESON_SUBDIR=${MESON_SUBDIR}"
echo "MESON_BUILD=${MESON_BUILD_ROOT}"

# (cd gsl-2.6 && 
#     ./configure --prefix=${MES} &&
#     make -j4 clean install
# )
# rm -rf gsl-2.6
exit 0
