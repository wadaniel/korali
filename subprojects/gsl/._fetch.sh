#!/bin/bash
VERSION="${1}"; shift
tarball="gsl-${VERSION}.tar.gz"
if [[ ! -f ${tarball} ]]; then
    git clean -xdf .
    wget "ftp://ftp.gnu.org/gnu/gsl/${tarball}"
    tar --strip-components=1 -xzvf ${tarball}
fi
