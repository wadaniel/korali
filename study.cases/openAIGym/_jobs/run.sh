#!/bin/bash -l
#SBATCH --job-name="OpenAI_VRACER_Humanoid-v2"
#SBATCH --output=OpenAI_%j.out
#SBATCH --error=OpenAI_%j.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-core=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=s1160

RUNPATH=/cluster/scratch/wadaniel/OpenAI/Humanoid-v2/$SLURM_JOB_ID
mkdir -p $RUNPATH

pushd ..

cat run-vracer.py

cp run-vracer.py $RUNPATH
cp settings.sh $RUNPATH
cp -r _model/ $RUNPATH

popd

pushd $RUNPATH

OMP_NUM_THREADS=12 python3 run-vracer.py --env "Humanoid-v2" --dis "Clipped Normal" --l2 0.0 --opt 0.1 --lr 0.0001

resdir=$(ls -d _result_vracer_*)
DISNW="$(echo -e "Clipped Normal" | tr -d '[:space:]')"
figureName=vracer_Humanoid-v2_${DISNW}_0.0_0.1_0.0001.png
python3 -m korali.rlview --dir $resdir --output $figureName

popd

date
