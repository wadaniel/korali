#!/bin/bash

run=15
noise=0.0
obj="random"
#dims=(2 4 8 16)
#dims=(32 64 128)
#pops=(6 8 10 12)
#pops=(14 16 18)
exp=3000000

#dims=(2 4 8 16 32 64)
#pops=(8 16 32 64 128 256)
dims=(2 64)
pops=(8 256)
for i in "${!dims[@]}";
do
    python run-vracer.py --noise $noise --obj $obj --dim ${dims[i]} --pop ${pops[i]} --run $run --exp $exp;  
    python run-vracer.py --noise $noise --obj $obj --dim ${dims[i]} --pop ${pops[i]} --run $run --eval; 
    python -m korali.rlview --dir "_vracer_${obj}_${dims[i]}_${pops[i]}_${noise}_${run}/" --out "${obj}_${dims[i]}_${pops[i]}_${run}.png"
done;

