#!/bin/bash

run=19
noise=0.0
reps=100
version=1
obj="random"
#dims=(2 4 8 16)
#dims=(32 64 128)
#pops=(6 8 10 12)
#pops=(14 16 18)
exp=3000000

#dims=(2 4 8 16 32 64)
#pops=(8 16 32 64 128 256)
dims=(32 64)
pops=(128 256)
#dims=(4)
#pops=(16)

objectives=("fsphere" "felli" "fcigar" "ftablet" "fcigtab" "ftwoax" "fdiffpow")

for i in "${!dims[@]}";
do
    python run-vracer.py --noise $noise --obj $obj --dim ${dims[i]} --pop ${pops[i]} --run $run --exp $exp --version=$version
    python -m korali.rlview --dir "_vracer_${obj}_${dims[i]}_${pops[i]}_${noise}_${run}/" --out "${obj}_${dims[i]}_${pops[i]}_${run}.png"
    
    python run-vracer.py --noise $noise --obj $obj --dim ${dims[i]} --pop ${pops[i]} --run $run --eval --reps $reps --version=$version

    for o in "${objectives[@]}";
    do
        python run-env-cmaes.py --noise $noise --obj $o --dim ${dims[i]} --pop ${pops[i]} --run $run --eval --reps $reps; 
        vracerfile="history_vracer_${o}_${dims[i]}_${pops[i]}_${noise}_${run}.npz"
        cmaesfile="history_cmaes_${o}_${dims[i]}_${pops[i]}_${noise}_${run}.npz"
        python read-history.py --files ${vracerfile} ${cmaesfile}
    done;

done;

