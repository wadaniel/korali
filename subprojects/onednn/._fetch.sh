#!/bin/bash
if [[ $# -ne 2 ]]; then
  echo "Usage: ${0} version target_dirctory"
  exit 1
fi

VERSION="${1}";
TARGET_DIR="${2}";

cd $TARGET_DIR

tarball="v${VERSION}.tar.gz"

if [[ ! -f ${tarball} ]]; then
    git clean -xdf .
    cp .gitignore .gitignore.bak
    curl -L -o ${tarball} "https://github.com/oneapi-src/oneDNN/archive/${tarball}"
    tar --strip-components=1 -xzvf ${tarball}
    mv .gitignore.bak .gitignore
fi
