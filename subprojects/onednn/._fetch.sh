#!/bin/bash
VERSION="${1}"; shift

tarball="v${VERSION}.tar.gz"

if [[ ! -f ${tarball} ]]; then
    git clean -xdf .
    wget "https://github.com/oneapi-src/oneDNN/archive/${tarball}"
    tar --strip-components=1 -xzvf ${tarball}
fi