#!/usr/bin/env bash

# Test plotting scripts

echo "----------------------------------------------"
echo "[Korali] Beginning plotting tests"
echo "----------------------------------------------"

pushd ../../examples/  > /dev/null
exit_code=$?

resDirs=`find . -name "_korali_result*"`

for dir in $resDirs
do
  echo "----------------------------------------------"
  echo "[Korali] Plotting results from $dir ..."
  echo "----------------------------------------------"
  python3 -m korali.plotter --test --dir "${dir}"
  exit_code=$(( $exit_code || $? ))
done

popd  > /dev/null
exit_code=$(( $exit_code || $? ))

exit $exit_code
