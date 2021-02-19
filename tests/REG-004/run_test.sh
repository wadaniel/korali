#!/usr/bin/env bash

if [ $# -gt 0 ]; then
  cd $1
fi

# Execute Solver Scripts

echo "----------------------------------------------"
echo "[Korali]  Beginning solver tests"

for file in *.py
do
  echo "----------------------------------------------"
  echo "[Korali] Running File: ${file%.*}"

  python3 ./$file
  exit_code=$?

  echo "[Korali] Removing results..."
  rm -rf "_korali_result"
  exit_code=$(( $exit_code || $? ))
done

exit $exit_code