#!/usr/bin/env bash

if [ $# -gt 0 ]; then
  cd $1
fi

echo "-------------------------------------"
echo "[Korali] Beginning Stat Test 005"
exit_code=$?

echo "-------------------------------------"
echo "Testing Maximizing"
echo "Running File: run-maxcmaes1.py"

python3 ./run-maxcmaes1.py
exit_code=$(( $exit_code || $? ))

echo "[Korali] Removing results..."
rm -rf "_korali_result"
exit_code=$(( $exit_code || $? ))

echo "-------------------------------------"

echo "-------------------------------------"
echo "Testing Maximizing"
echo "Running File: run-maxcmaes2.py"

python3 ./run-maxcmaes2.py
exit_code=$(( $exit_code || $? ))

echo "[Korali] Removing results..."
rm -rf "_korali_result"
exit_code=$(( $exit_code || $? ))

echo "-------------------------------------"

echo "-------------------------------------"
echo "Testing Minimizing"
echo "Running File: run-mincmaes1.py"

python3 ./run-mincmaes1.py
exit_code=$(( $exit_code || $? ))

echo "[Korali] Removing results..."
rm -rf "_korali_result"
exit_code=$(( $exit_code || $? ))

echo "-------------------------------------"

echo "-------------------------------------"
echo "Testing Minimizing"
echo "Running File: run-mincmaes2.py"

python3 ./run-mincmaes2.py
exit_code=$(( $exit_code || $? ))

echo "[Korali] Removing results..."
rm -rf "_korali_result"
exit_code=$(( $exit_code || $? ))

echo "-------------------------------------"

echo "-------------------------------------"
echo "Testing Minimizing C++"
echo "Build & Running File: run-mincmaes1.cpp"

make
exit_code=$(( $exit_code || $? ))

python3 ./run-mincmaes2.py
exit_code=$(( $exit_code || $? ))

echo "[Korali] Removing results..."
rm -rf "_korali_result"
exit_code=$(( $exit_code || $? ))

make clean
echo "-------------------------------------"

exit $exit_code