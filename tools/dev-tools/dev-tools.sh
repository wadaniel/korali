#!/bin/bash
##### Automatically downloads all prerequisites for Korali developers

# Getting common tools
source ../../tests/functions.sh

# Step 1: Get Sphinx and ReadTheDocs

python3 -m pip install sphinx --user
check_result

python3 -m pip install sphinx_rtd_theme --user
check_result

python3 -m pip install Pygments --user
check_result

# Step 2: Doxygen

if [ -f ../../build/doxygen ]; then
 echo "[Korali] Seems like you already have doxygen installed ../../build/doxygen. Skipping..."
else

 rm -rf doxygen
 check_result
 
 git clone https://github.com/doxygen/doxygen.git
 check_result
 
 cd doxygen
 check_result
 
 mkdir build
 check_result
 
 pushd build
 check_result
 
 cmake -G "Unix Makefiles" ..
 check_result
 
 make -j4 
 check_result
 
 cp bin/doxygen ../../build/doxygen 
 check_result
 
 popd
 check_result
fi

# Step 3: Clang Tools

popd 
