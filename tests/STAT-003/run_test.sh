#!/usr/bin/env bash

echo "-------------------------------------"
echo "[Korali] Beginning Stat Test 002"
echo "[Korali] minimization with non-finites tests..."
exit_code=$?

for file in run*.py
do
  echo "-------------------------------------"
  echo " Running $file"
  echo "-------------------------------------"
  ./"$file"
  exit_code=$(( $exit_code || $? ))
done

echo "[Korali] Removing results..."
rm -rf _results_*
exit_code=$(( $exit_code || $? ))

exit $exit_code