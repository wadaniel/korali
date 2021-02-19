#!/usr/bin/env bash

if [ $# -gt 0 ]; then
  cd $1
fi

echo "-------------------------------------"
echo "[Korali] Beginning Stat Test 001"
exit_code=$?

echo $(pwd)

for file in $(find . -name "run*.py")
do
  echo "-------------------------------------"
  echo " Running $file"
  echo "-------------------------------------"
  ./"$file"
  exit_code=$(( $exit_code || $? ))
done

echo "[Korali] Removing results..."
rm -rf _result_run-*
exit_code=$(( $exit_code || $? ))

exit $exit_code