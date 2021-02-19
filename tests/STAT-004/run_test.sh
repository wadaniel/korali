#!/usr/bin/env bash

if [ $# -gt 0 ]; then
  cd $1
fi

echo "-------------------------------------"
echo "[Korali] Beginning Stat Test 004"
exit_code=$?

for file in run-*.py
do
  echo "-------------------------------------"
  echo " Running $file"
  echo "-------------------------------------"
  ./"$file"
  exit_code=$(( $exit_code || $? ))
done

echo "[Korali] Removing results..."
rm -rf _korali_result
exit_code=$(( $exit_code || $? ))

exit $exit_code