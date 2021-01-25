#!/bin/bash
export OMP_NUM_THREADS=2

for i in {1..10}
do
    fname=run-vracer-t0-$i.py
    sed "5 a run = $i\ntarget = 0.0" run-vracer.py > $fname
    python3 $fname
done
