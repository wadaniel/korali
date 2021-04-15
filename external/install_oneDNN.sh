#!/bin/bash

######### Global Definitions ########
libName="oneDNN"
binName="oneDNN"
minVersion=2.1.1

######### Helper Functions ########

function check()
{
 if [ ! $? -eq 0 ]
 then
  exit $?
 fi 
}

######### Environment Configuration ########

baseKoraliDir=$PWD
foundbinVersionFile=0
source ${baseKoraliDir}/install.config

if [ -f $baseKoraliDir/docs/VERSION ]; then
 foundbinVersionFile=1
fi

if [ -f $baseKoraliDir/../docs/VERSION ]; then
 foundbinVersionFile=1
 baseKoraliDir=`dirname $PWD`
fi

if [ $foundbinVersionFile == 0 ]; then
  echo "[Korali] Error: You need to run this file from Korali's base folder."
  exit 1
fi

######### Looking for CMake ##########

cmakeMajorVersion=`$CMAKE --version | grep "cmake version" | cut -d' ' -f 3 | cut -d '.' -f 1`
if [ ! $? == 0 ]; then
 logEcho "[Korali] Error: Could not find a valid CMake (path: $CMAKE)"
 logEcho "[Korali] Solution: Provide a path to a correct version in the 'install.config' file."
 exitWithError
fi

if [ "$cmakeMajorVersion" -lt "3" ]; then
 logEcho "[Korali] Error: CMake found is older than version than 3.0."
 logEcho "[Korali] Solution: Provide a path to a correct version in the 'install.config' file."
 exitWithError
fi

######## Checking for existing software ########

externalDir=${baseKoraliDir}/external

baseLibDir=${externalDir}/${libName}
installDir=${baseLibDir}/install
buildDir=${baseLibDir}/build

fileFound=0
localFile=${installDir}/include/dnnl.hpp
globalFile=${DNNLROOT}/include/dnnl.hpp

if [ -f ${globalFile} ]; then
 fileFound=1
 filePath=${globalFile}
 oneDNNBaseDir=${DNNLROOT}
fi 

if [ -f ${localFile} ]; then
 fileFound=1
 filePath=${localFile}
 oneDNNBaseDir=$installDir
fi

######## If not installed, download and install ########

if [ ${fileFound} == 0 ]; then

 echo "[Korali] Downloading ${libName}... "
 
 rm -rf $buildDir; check
 rm -rf $installDir; check

 mkdir -p $buildDir; check
 pushd $buildDir; check
 
 dst="v${minVersion}"
 rm -f ${dst}.tar.gz; check
 rm -rf ${dst}; check
 
 wget https://github.com/oneapi-src/oneDNN/archive/${dst}.tar.gz; check
 tar -xzvf ${dst}.tar.gz ; check
  
 echo "[Korali] Configuring ${libName}... "
 cd "oneDNN-${minVersion}"
 mkdir -p build; check
 cd build; check

 CXXFLAGS=-O3 ${CMAKE} .. \
     -DDNNL_BUILD_EXAMPLES=OFF \
     -DDNNL_BUILD_TESTS=OFF \
     -DDNNL_ENABLE_CONCURRENT_EXEC=ON \
     -DCMAKE_INSTALL_PREFIX=${installDir} \
     -DDNNL_ARCH_OPT_FLAGS='-march=native -mtune=native' \
     -DBUILD_SHARED_LIBS=true; check

 echo "[Korali] Building ${libName}... "
 make -j$NJOBS; check
 
 echo "[Korali] Installing ${libName}... "
 make install; check
 
 popd; check
 
 echo "[Korali] Finished installing ${libName}."
 binPath=${installDir}/bin/${binName}
 
 echo "[Korali] Cleaning up build folder..."
 rm -rf $buildDir; check
 
 oneDNNBaseDir=$installDir
fi

######## Finalization ########

rm -f ${externalDir}/${binName}link
ln -sf ${oneDNNBaseDir} ${externalDir}/${binName}link; check
echo "[Korali] Using oneDNN located at ${oneDNNBaseDir}."

exit 0
