#!/bin/bash

function check()
{
 if [ ! $? -eq 0 ]
 then
  echo "[Korali] Error obtaining developer tools."
  exit -1
 fi 
}

##### Automatically downloads all prerequisites for Korali developers

# Getting common tools
source ../../tests/functions.sh

# Step 1: Doxygen

if [ -f doxygen ]; then
 echo "[Korali] Seems like you already have doxygen installed. Skipping..."
else

 rm -rf doxygen-src
 check
 
 git clone https://github.com/doxygen/doxygen.git doxygen-src
 check
 
 mkdir doxygen-src/build
 check
 
 pushd doxygen-src/build
 check 

 cmake -G "Unix Makefiles" ..
 check
 
 make -j4 
 check
 
 popd
 check
 
 cp doxygen-src/build/bin/doxygen .
 check 

 rm -rf doxygen-src
 check
fi

# Step 2: Clang Tools

if [ -f clang-format ]; then
 echo "[Korali] Seems like you already have clang-format installed. Skipping..."
else

 # If using MacOs, use the Darwin package 
 if [ "$arch" == "Darwin" ]; then
  
  wget https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.0/clang+llvm-10.0.0-x86_64-apple-darwin.tar.xz
  check
  
  tar --extract --file=clang+llvm-10.0.0-x86_64-apple-darwin.tar.xz clang+llvm-10.0.0-x86_64-apple-darwin/bin/clang-format --strip=2
  check
  
 else  # Else default to Linux64
 
  wget https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.0/clang+llvm-10.0.0-x86_64-linux-sles11.3.tar.xz
  check
  
  tar --extract --file=clang+llvm-10.0.0-x86_64-linux-sles11.3.tar.xz clang+llvm-10.0.0-x86_64-linux-sles11.3/bin/clang-format --strip=2
  check
  
 fi
 
fi

# Delete any remaining files
rm *.xz
check

