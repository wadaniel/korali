#!/usr/bin/env bash

# Brief: Re-run all examples and features for basic sanity check.

echo "----------------------------------------------"
echo "[Korali] Beginning examples test..."
echo "----------------------------------------------"

dir=$PWD/../../examples

pushd $dir  > /dev/null
exit_code=$?

./.run_test.sh
exit_code=$(( $exit_code || $? ))

popd > /dev/null
exit_code=$(( $exit_code || $? ))


exit $exit_code