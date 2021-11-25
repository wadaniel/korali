#!/bin/bash -l

ENV="Swimmer-v2"
DIS="Clipped Normal"
LRS=0.0005
BATCH=512
#512
EPOCH=300
LAYERS=5
#5
UNITS=300
#512
P=0.0
SIZE=2500000
#65526
WS=1.0
#OUTSCA="MinMax"
echo $ENV
echo $DIS
echo $LRS
echo $BATCH
echo $EPOCH
echo $LAYERS
echo $UNITS
echo $P
echo $SIZE
echo $WS
#echo $OUTSCA

cat > rungs.sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name="OpenAI_grid_search_$ENV"
#SBATCH --output=OpenAI_grid_search_$ENV_%j.out
#SBATCH --error=OpenAI_grid_search_$ENV_err_%j.out
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-core=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=s929

RUNPATH=$SCRATCH/OpenAI_grid_search/$ENV/\$SLURM_JOB_ID
mkdir -p \$RUNPATH

pushd ..

#cat _model_openAIgym/model.py

cp run-vracer-openAIgym.py \$RUNPATH
cp settings_openAIgym.sh \$RUNPATH
cp -r _model_openAIgym/ \$RUNPATH

popd

pushd \$RUNPATH/_model_openAIgym

OMP_NUM_THREADS=12 srun python3 model.py --env "$ENV" --dis "$DIS" --lr $LRS --batch $BATCH --epoch $EPOCH --layers $LAYERS --units $UNITS --p $P --size $SIZE --ws $WS
# --outputScaler "$OUTSCA"

popd

date
EOF

sbatch rungs.sbatch

