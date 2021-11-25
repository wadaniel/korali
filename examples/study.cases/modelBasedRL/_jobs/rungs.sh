#!/bin/bash -l
#SBATCH --job-name="OpenAI_grid_search_Swimmer-v2"
#SBATCH --output=OpenAI_grid_search_%j.out
#SBATCH --error=OpenAI_grid_search_%j.out
#SBATCH --time=13:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-core=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=s929

RUNPATH=/scratch/snx3000/jclapesr/OpenAI_grid_search/Swimmer-v2
mkdir -p $RUNPATH

pushd ..

#cat model_openAIgym/model.py

cp run-vracer-openAIgym.py $RUNPATH
cp settings_openAIgym.sh $RUNPATH
cp -r _model_openAIgym/ $RUNPATH

popd

pushd $RUNPATH/_model_openAIgym

OMP_NUM_THREADS=12 srun python3 model.py --env "Swimmer-v2" --dis "Clipped Normal" --lr 0.0005 --batch 512 --epoch 250 --layers 3 --units 100 --p 0.0 --size 2500000 --ws 0.75

popd

date
