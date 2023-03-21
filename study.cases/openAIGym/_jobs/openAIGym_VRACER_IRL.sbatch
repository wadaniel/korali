#!/bin/bash -l

#SBATCH --job-name="openai_irl"
#SBATCH --output=openai_irl_%j.out
#SBATCH --error=openai_irl_err_%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-core=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=s1160
#SBATCH --mail-user=wadaniel@ethz.ch
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

RUNPATH=$SCRATCH/openai_irl_march/$ENV/$SLURM_JOB_ID
mkdir -p $RUNPATH

pushd ..

export RUN=-1
cat run-vracer-irl.py
source settings_irl.sh

cp run-vracer-irl.py $RUNPATH
cp settings_irl.sh $RUNPATH
cp -r _model/ $RUNPATH
cp observations*${ENV}* $RUNPATH

touch "${RUNPATH}/run.config"
echo ENV    ${ENV} >> "${RUNPATH}/run.config"
echo EBRU   ${EBRU} >> "${RUNPATH}/run.config"
echo DBS    ${DBS} >> "${RUNPATH}/run.config"
echo BBS    ${BBS} >> "${RUNPATH}/run.config"
echo BSS    ${BSS} >> "${RUNPATH}/run.config"
echo EXP    ${EXP} >> "${RUNPATH}/run.config"
echo RNN    ${RNN} >> "${RUNPATH}/run.config"
echo POL    ${POL} >> "${RUNPATH}/run.config"
echo RUN    ${RUN} >> "${RUNPATH}/run.config"

popd

pushd $RUNPATH

OMP_NUM_THREADS=12 python3 run-vracer-irl.py --env ${ENV} --ebru ${EBRU} --dbs ${DBS} --bbs ${BBS} --bss ${BSS} --exp ${EXP} --rnn ${RNN} --pol ${POL} --run ${RUN} 

DIR="${BASE}/_result_irl_${ENV}_${RUN}/"
python3 -m korali.plot --dir $DIR --out "${ENV}_${RUN}.png"
python3 -m korali.rlview --dir $DIR --out "irl-${ENV}_${RUN}.png" --showObservations --showCI 0.8
python3 -m korali.rlview --dir $DIR --out "firl-${ENV}_${RUN}.png" --featureReward --showObservations --showCI 0.8


popd

date
