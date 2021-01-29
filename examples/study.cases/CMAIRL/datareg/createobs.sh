#!/bin/bash

# script to produce observations and policies

export OMP_NUM_THREADS=4

#target=0.0
#target=0.0625
target=0.125
#target=0.25
#target=0.5
#target=1.0

RUNDIR="./t${target}" # required to place copy of korali app

mkdir $RUNDIR
for i in {1..5}
do
    fname=run-vracer-t${target}-$i.py
    sed "5 a run = $i\ntarget = ${target}" run-vracer.py > "${RUNDIR}/${fname}"
    pushd .
    cd $RUNDIR
    python3 $fname
    popd
done
