#!/usr/bin/env bash

if [ $# -gt 0 ]; then
  cd $1
fi

<<<<<<< HEAD:examples/learning/supervised/sineWave/.run_test.sh
source ../../../../tests/functions.sh

##### Deleting Previous Results 
=======
##### Deleting Previous Results
>>>>>>> 9a5896b8019c72d79425909e1c394a3dba9aa906:examples/learning/supervised/.run_test.sh

echo "  + Deleting previous results..."
rm -rf _korali_result*
exit_code=$?

##### Creating test files

echo "  + Creating test files..."

<<<<<<< HEAD:examples/learning/supervised/sineWave/.run_test.sh
sed -e 's%\plt.%#plt.%g' \
        run-ffn.py > __test-ffn.py; check_result

##### Running Test

OMP_NUM_THREADS=4 python3 ./__test-ffn.py; check_result
=======
sed -e 's%\plt.%#plt.%g' run-sin.py > __test-sin.py
exit_code=$(( $exit_code || $? ))

##### Running Test

OMP_NUM_THREADS=4 python3 ./__test-sin.py
exit_code=$(( $exit_code || $? ))
>>>>>>> 9a5896b8019c72d79425909e1c394a3dba9aa906:examples/learning/supervised/.run_test.sh

##### Deleting Tests

#rm -rf __test-*
exit_code=$(( $exit_code || $? ))


exit $exit_code
