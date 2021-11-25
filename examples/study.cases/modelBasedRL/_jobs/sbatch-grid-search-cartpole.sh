#!/bin/bash -l

echo $LRS
echo $BATCH
echo $EPOCH
echo $LAYERS
echo $UNITS
echo $P
echo $SIZE
echo $WS

cat > rungs.sh <<EOF
#!/bin/bash -l
#SBATCH --job-name="cartpole_grid_search"
#SBATCH --output=cartpole_grid_search_%j.out
#SBATCH --error=cartpole_grid_search_%j.out
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-core=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=s929

RUNPATH=$SCRATCH/Cartpole_grid_search
mkdir -p \$RUNPATH

pushd ..

#cat model_cartpole/model.py

cp run-vracer-cartpole.py \$RUNPATH
cp -r _model_cartpole/ \$RUNPATH

popd

pushd \$RUNPATH/_model_cartpole

OMP_NUM_THREADS=12 srun python3 model.py --batch $BATCH --epoch $EPOCH --layers $LAYERS --units $UNITS --p $P --size $SIZE --ws $WS

popd

date
EOF

sbatch rungs.sh
