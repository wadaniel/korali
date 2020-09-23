#!/bin/bash

######### Global Definitions ########
libName="eigen"
minVersion=3.3.7

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
installDir=${baseLibDir}/install/
buildDir=${baseLibDir}/build

fileFound=0
localFile=${installDir}/include/eigen3/Eigen/Core

if [ -f ${localFile} ]; then
 fileFound=1
 filePath=${localFile}
fi
 
######## If not installed, download and install ########

if [ ${fileFound} == 0 ]; then
 
 echo "[Korali] Downloading ${libName}... "
 
 rm -rf $buildDir; check
 rm -rf $installDir; check
 
 mkdir -p $buildDir; check
 mkdir -p $installDir; check
 pushd $buildDir; check
 
 dst="eigen-${minVersion}"
 rm -f ${dst}.tar.gz; check
 rm -rf ${dst}; check
 
 wget https://gitlab.com/libeigen/eigen/-/archive/${minVersion}/${dst}.tar.gz; check
 tar -xzvf ${dst}.tar.gz ; check
  
 echo "[Korali] Configuring ${libName}... "
 cd ${dst}
 mkdir -p build; check
 cd build; check

 CXXFLAGS=-O3 ${CMAKE} .. \
     -DBUILD_TESTING=OFF \
     -DCMAKE_INSTALL_PREFIX=${installDir}; check

 echo "[Korali] Building ${libName}... "
 make -j$NJOBS; check
 
 echo "[Korali] Installing ${libName}... "
 make install; check
 
 popd; check
 
 echo "[Korali] Finished installing ${libName}."
 binPath=${installDir}/bin/${binName}
 
 echo "[Korali] Cleaning up build folder..."
 rm -rf $buildDir; check
 
fi

######## Finalization ########

rm -f ${externalDir}/eigenlink
ln -sf ${installDir} ${externalDir}/eigenlink; check
echo "[Korali] Using Eigen located at ${installDir}."

exit 0
