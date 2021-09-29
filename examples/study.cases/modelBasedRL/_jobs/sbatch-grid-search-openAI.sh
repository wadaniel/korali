#!/bin/bash -l

source ../settings_openAIgym.sh

echo $ENV
echo $DIS
echo $LR
echo $BATCH
echo $EPOCH
echo $LAYERS
echo $UNITS
echo $P
echo $SIZE
echo $WS

cat > rungs.sh <<EOF
#!/bin/bash -l
#SBATCH --job-name="OpenAI_grid_search_$ENV"
#SBATCH --output=OpenAI_grid_search_$ENV_%j.out
#SBATCH --error=OpenAI_grid_search_$ENV_err_%j.out
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-core=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=s929

RUNPATH=$SCRATCH/OpenAI_grid_search/$ENV
mkdir -p \$RUNPATH

pushd ..

cat model_openAIgym/model.py

cp run-vracer-openAIgym.py \$RUNPATH
cp settings_openAIgym.sh \$RUNPATH
cp -r _model_openAIgym/ \$RUNPATH

popd

pushd \$RUNPATH/_model_openAIgym

OMP_NUM_THREADS=12 python3 model.py --env "$ENV" --dis "$DIS" --lr $LR --batch $BATCH --epoch $EPOCH --layers $LAYERS --units $UNITS --p $P --size $SIZE --ws $WS

popd

date
EOF

sbatch rungs.sh
