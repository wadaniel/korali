#!/usr/bin/env bash

# Check Code Formatting and documentation

echo "----------------------------------------------"
echo "[Korali] Beginning code formatting check..."
echo "----------------------------------------------"

pushd ../../tools/style/  > /dev/null
exit_code=$?

./check_style_cxx.sh
exit_code=$(( $exit_code || $? ))

popd  > /dev/null
exit_code=$(( $exit_code || $? ))


echo "----------------------------------------------"
echo "[Korali] Beginning code formatting check..."
echo "----------------------------------------------"

pushd ../../docs/  > /dev/null
exit_code=$(( $exit_code || $? ))

./build.sh
exit_code=$(( $exit_code || $? ))

popd  > /dev/null
exit_code=$(( $exit_code || $? ))

exit $exit_code