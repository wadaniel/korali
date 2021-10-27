#!/bin/bash

pushd ../..

nLinesDiff=`git --no-pager diff source | wc -l`
if [ $nLinesDiff -ne 0 ]; then
 echo "[Korali] Error: Found differences in modules .cpp and/or .hpp files after build. This means that you have changed a .cpp/.hpp directly instead of the .base file. Another cause is that you have changed the .base, but did not re-build before committing. Please fix your commit such that .base and code files correspond each other after build." 
 echo "[Korali] Diff Details:"
 git --no-pager diff source
 exit 1
fi
