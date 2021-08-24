#!/bin/bash

run=9
noise=0.0
obj="random"
dims=(2 4 8 16 32 64 128)
pops=(6 8 10 12 14 16 18)

for i in "${!dims[@]}";
do
    python run-vracer.py --noise $noise --obj $obj --dim ${dims[i]} --pop ${pops[i]} --run $run; 
    python run-vracer.py --noise $noise --obj $obj --dim ${dims[i]} --pop ${pops[i]} --run $run --eval; 
    python -m korali.rlview --dir "_vracer_${obj}_${dims[i]}_${pops[i]}_${noise}_${run}/" --out "${obj}_${dims[i]}_${pops[i]}_${run}.png"

done;

