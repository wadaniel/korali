#!/bin/bash

function check()
{
 if [ ! $? -eq 0 ]
 then
  echo "[Korali] Error obtaining developer tools."
  exit -1
 fi 
}

##### Detecting architecture

arch="$(uname -s)"
check

##### Automatically downloads all prerequisites for Korali developers


# Step 1: CMake

which cmake > /dev/null 2>&1
if [ $? -eq 0 ] || [ -f cmake/bin/cmake ]; then
 echo "[Korali] Seems like you already have CMake installed. Skipping..."
else

 # If using MacOs, use the Darwin package 
 if [ "$arch" == "Darwin" ]; then
  
  wget https://github.com/Kitware/CMake/releases/download/v3.17.2/cmake-3.17.2-Darwin-x86_64.tar.gz
  check
  
  tar -xzvf cmake-3.17.2-Darwin-x86_64.tar.gz
  check
  
  mv cmake-3.17.2-Darwin-x86_64 cmake
  check
  
 else  # Else default to Linux64
 
  wget https://github.com/Kitware/CMake/releases/download/v3.17.2/cmake-3.17.2-Linux-x86_64.tar.gz
  check
  
  tar -xzvf cmake-3.17.2-Linux-x86_64.tar.gz
  check
  
 fi
 
 mv cmake-3.17.2-Linux-x86_64 cmake
 check
 
fi

# Step 2: Doxygen

which doxygen > /dev/null 2>&1
if [ $? -eq 0 ] || [ -f doxygen ]; then
 echo "[Korali] Seems like you already have doxygen installed. Skipping..."
else

 # If using MacOs, use the Darwin package 
 if [ "$arch" == "Darwin" ]; then
  
  wget http://doxygen.nl/files/Doxygen-1.8.18.dmg
  check
  
  hdiutil attach Doxygen-1.8.18.dmg
  check
  
  cp /Volumes/Doxygen/Doxygen.app/Contents/Resources/doxygen .
  check
  
 else  # Else default to Linux64
 
  wget http://doxygen.nl/files/doxygen-1.8.18.linux.bin.tar.gz
  check
  
  tar -xzvf doxygen-1.8.18.linux.bin.tar.gz
  check
  
  cp doxygen-1.8.18/bin/doxygen .
  check
  
 fi
fi

# Step 3: Clang Tools

which clang-format > /dev/null 2>&1
if [ $? -eq 0 ] || [ -f clang-format ]; then
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
rm -rf *.xz
rm -rf *.tar.gz
rm -rf doxygen-*
check
