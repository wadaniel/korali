#!/bin/bash

######### Global Definitions ########
libName="CMake"
binName="cmake"

######### Helper Functions ########

function check()
{
 if [ ! $? -eq 0 ]
 then
  exitWithError
 fi 
}

# Help display function
function print_help ()
{
 echo ""
 echo "Syntax: ./install_cmake.sh [--jobs=N]"
 echo ""
 echo "Where:"
 echo " --jobs=N Specifies N jobs to use when building cmake."
 echo " --help Displays this help message."
}

######### Environment Configuration ########

NJOBS=4
baseKoraliDir=$PWD
foundVersionFile=0

if [ -f $baseKoraliDir/docs/VERSION ]; then
 foundVersionFile=1
fi

if [ -f $baseKoraliDir/../docs/VERSION ]; then
 foundVersionFile=1
 baseKoraliDir=$PWD/../
fi

if [ $foundVersionFile == 0 ]; then
  echo "[Korali] Error: You need to run this file from Korali's base folder."
  exit -1
fi

echo $NJOBS
exit 0

######### Argument Parsing ########

for i in "$@"
do
case $i in
    --jobs=*)
    NJOBS="${i#*=}"
    shift
    ;;
    --help)
    print_help
    exit 0 
    shift 
    ;;
    *)
    print_help
    echo ""
    echo "[Korali] Error:  Unknown option $i."       
    exit 1
    ;;
esac
done

######## Checking for existing software ########

installDir=${baseKoraliDir}/prereqs/cmakeInstall/
buildDir=${baseKoraliDir}/prereqs/cmakeBuild/

isFound=0
version=0.0
binary=${installDir}/bin/cmake

if [ ! -f ${binary} ]; then
 binary=cmake
fi

$binary --version > /dev/null 2>&1
if [ $? -eq 0 ]; then
 minversion=3.0
 version=`cmake --version | head -n 1 | cut -d' ' -f 3`
 cmpver=`printf "${version}\n${minversion}" | sort -V | head -n 1`
 
 if [[ "$cmpver" != "$minversion" ]]; then
    echo "[Korali] CMake version found (${version}) is smaller than required (${minversion}). Installing newer version..."
 else
    isFound=1
    echo "[Korali] Found existing CMake version ${version}. Skipping installation..."
 fi
fi

######## If not installed, download and install ########

if [ ${isFound} == 0 ]; then

 echo "[Korali] Downloading CMake... "
 
 rm -rf $buildDir; check
 mkdir -p $buildDir; check
 pushd $buildDir; check
 
 rm -f cmake-3.17.3.tar.gz; check
 rm -rf cmake-3.17.3; check
 rm -rf cmakeInstall; check
 
 wget https://github.com/Kitware/CMake/releases/download/v3.17.3/cmake-3.17.3.tar.gz; check
 tar -xzvf cmake-3.17.3.tar.gz; check
  
 echo "[Korali] Configuring CMake... "
 cd cmake-3.17.3/
 ./configure --prefix=${CURDIR}/build/cmakeInstall --parallel=$NJOBS; check
 
 echo "[Korali] Building CMake... "
 make -j$NJOBS; check
 
 echo "[Korali] Installing CMake... "
 make install; check
 
 popd; check
 
 echo "[Korali] Finished installing CMake. "
 binary=${installDir}/bin/cmake
 
 echo "[Korali] Cleaning up build folder. "
 rm -rf $buildDir; check
fi

######## Finalization ########

version=`$binary --version | head -n 1 | cut -d' ' -f 3` 
echo "[Korali] Using CMake version $version"
