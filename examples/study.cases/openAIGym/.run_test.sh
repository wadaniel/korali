#!/bin/bash


###### Check OS ######

if [ "$(uname)" == "Darwin" ]; then
    exit
fi

###### Auxiliar Functions and Variables #########

source ../../../tests/functions.sh

###### Check if necessary python modules are installed ######

python3 -m pip install gym; check_result
./install_deps.sh; check_result

##### Deleting Previous Results 

echo "  + Deleting previous results..."
rm -rf _result*; check_result

##### Creating test files
 
echo "  + Creating test files..."

rm -rf __test-*; check_result

for file in *.py
do
 sed -e 's%\k.run(e)%e["Solver"]["Termination Criteria"]["Max Generations"] = 20; k.run(e)%g' ${file} > __test-${file}; check_result
done

##### Running Test

file="__test-run-vracer.py"
echo "Running ${file} ..."
OMP_NUM_THREADS=4 python3 ${file} --env AntBulletEnv-v0; check_result

#file="__test-genMovie.py"
#echo "Running ${file} ..."
#OMP_NUM_THREADS=4 python3 ${file} --env AntBulletEnv-v0 --input _result_vracer_AntBulletEnv-v0 --output movie; check_result

##### Deleting Tests

rm -rf __test-*; check_result
  
  
