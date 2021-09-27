#!/bin/bash -l

cat > run.sh <<EOF
#!/bin/bash -l
#SBATCH --job-name="RLCMAES_VRACER_$dim"
#SBATCH --output=RLCMAES_$dim_%j.out
#SBATCH --error=RLCMAES_$dim_err_%j.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-core=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=s929

RUNPATH=$SCRATCH/RLCMAES/$dim/\$SLURM_JOB_ID
mkdir -p \$RUNPATH

cat run-vracer.py

cp run-vracer.py \$RUNPATH
cp -r _environment/ \$RUNPATH

pushd \$RUNPATH

run=0
noise=0.0
obj="random"
exp=10000000
reps=100
version=1
outdir="figures"


#dims=(2 4 8 16 32 64)
#pops=(8 16 32 64 128 256)

#dims=(32)
#pops=(128)

dims=(64)
pops=(256)


objectives=("fsphere" "felli" "fcigar" "ftablet" "fcigtab" "ftwoax" "fdiffpow" "rosenbrock" "fparabr" "fsharpr")

mkdir -p ${outdir}

for i in "${!dims[@]}";
do
    python run-vracer.py --noise $noise --obj $obj --dim ${dims[i]} --pop ${pops[i]} --run $run --exp $exp --version=$version
    python -m korali.rlview --dir "_vracer_${obj}_${dims[i]}_${pops[i]}_${noise}_${run}/" --out "${outdir}/${obj}_${dims[i]}_${pops[i]}_${run}.png"
    
    python run-vracer.py --noise $noise --obj $obj --dim ${dims[i]} --pop ${pops[i]} --run $run --eval --reps $reps --version=$version

    for o in "${objectives[@]}";
    do
        python run-env-cmaes.py --noise $noise --obj $o --dim ${dims[i]} --pop ${pops[i]} --run $run --eval --reps $reps; 
        vracerfile="history_vracer_${o}_${dims[i]}_${pops[i]}_${noise}_${run}.npz"
        cmaesfile="history_cmaes_${o}_${dims[i]}_${pops[i]}_${noise}_${run}.npz"
        outfile="${outdir}/history_${o}_${dims[i]}_${pops[i]}_${noise}_${run}.png"
        python read-history.py --files ${vracerfile} ${cmaesfile} --out ${outfile}
    done;

done;

popd

date
EOF

sbatch run.sh
