#!/bin/bash

######### Global Definitions ########
libName="CMake"
binName="cmake"
minVersion=3.0

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
 echo "Syntax: ./install_${libName}.sh [--jobs=N]"
 echo ""
 echo "Where:"
 echo " --jobs=N Specifies N jobs to use when building ${libName}."
 echo " --help Displays this help message."
}

######### Environment Configuration ########

NJOBS=4
baseKoraliDir=$PWD
foundbinVersionFile=0

if [ -f $baseKoraliDir/docs/VERSION ]; then
 foundbinVersionFile=1
fi

if [ -f $baseKoraliDir/../docs/VERSION ]; then
 foundbinVersionFile=1
 baseKoraliDir=$PWD/../
fi

if [ $foundbinVersionFile == 0 ]; then
  echo "[Korali] Error: You need to run this file from Korali's base folder."
  exit -1
fi

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

baseLibDir=${baseKoraliDir}/prereqs/${libName}
installDir=${baseLibDir}/install
buildDir=${baseLibDir}/build

binFound=0
binVersion=0.0
binPath=${installDir}/bin/${binName}

#if [ ! -f ${binPath} ]; then
# binPath=${binName}
#fi

$binPath --version > /dev/null 2>&1
if [ $? -eq 0 ]; then
 binVersion=`${binPath} --version | head -n 1 | cut -d' ' -f 3`
 cmpver=`printf "${binVersion}\n${minVersion}" | sort -V | head -n 1`
 
 if [[ "$cmpver" != "$minVersion" ]]; then
    echo "[Korali] ${libName} version found (${binVersion}) is smaller than required (${minVersion}). Installing newer version..."
 else
    binFound=1
    echo "[Korali] Found existing ${libName} version ${binVersion}. Skipping installation..."
 fi
fi

######## If not installed, download and install ########

if [ ${binFound} == 0 ]; then

 echo "[Korali] Downloading ${libName}... "
 
 rm -rf $buildDir; check
 rm -rf $installDir; check
 
 mkdir -p $buildDir; check
 pushd $buildDir; check
 
 rm -f cmake-3.17.3.tar.gz; check
 rm -rf cmake-3.17.3; check
 
 wget https://github.com/Kitware/CMake/releases/download/v3.17.3/cmake-3.17.3.tar.gz; check
 tar -xzvf cmake-3.17.3.tar.gz; check
  
 echo "[Korali] Configuring ${libName}... "
 cd cmake-3.17.3/
 ./configure --prefix=$installDir --parallel=$NJOBS; check
 
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

binVersion=`$binPath --version | head -n 1 | cut -d' ' -f 3` 
echo "[Korali] Using ${libName} version $binVersion"
