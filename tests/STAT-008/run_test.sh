#!/usr/bin/env bash

echo "-------------------------------------"
echo "[Korali] Beginning Stat Test 008"
exit_code=$?

for file in *.py
do
  logEcho "-------------------------------------"
  logEcho "Running File: ${file%.*}"

  python3 ./$file
  exit_code=$(( $exit_code || $? ))

  log "[Korali] Removing results..."
  rm -rf "_korali_result"
  exit_code=$(( $exit_code || $? ))

  logEcho "-------------------------------------"
done

exit $exit_code