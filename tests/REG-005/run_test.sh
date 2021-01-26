#!/usr/bin/env bash

pushd ../../ > /dev/null

echo "----------------------------------------------"
echo "[Korali] Beginning profiling tests..."

profFiles=`find . -name profiling.json`
exit_code=$?

for f in $profFiles
do
   echo "----------------------------------------------"
   echo "[Korali] Processing profiler information from $f ..."
   echo "----------------------------------------------"
   echo "python3 -m korali.profiler --test --input $f "
   exit_code=$(( $exit_code || $? ))
done

popd > /dev/null
exit_code=$(( $exit_code || $? ))

exit $exit_code