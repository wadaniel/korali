#!/bin/bash -l

#SBATCH --job-name="openai_irl"
#SBATCH --output=openai_irl_%j.out
#SBATCH --error=openai_irl_err_%j.out
##SBATCH --time=24:00:00
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-core=1
##SBATCH --partition=normal
#SBATCH --partition=debug
#SBATCH --constraint=gpu
#SBATCH --account=s1160
#SBATCH --mail-user=wadaniel@ethz.ch
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

mkdir -p $RUNPATH

cd ..
export OMP_NUM_THREADS=12 
source settings_irl.sh

RUNPATH="${SCRATCH}/openai_irl_march/${ENV}/${SLURM_JOB_ID}"
DIR="${RUNPATH}/_result_irl_${ENV}_${RUN}/"

cat run-vracer-irl.py

mkdir ${DIR} -p
touch "${DIR}/run.config"
echo ENV    ${ENV} >> "${DIR}/run.config"
echo EBRU   ${EBRU} >> "${DIR}/run.config"
echo DBS    ${DBS} >> "${DIR}/run.config"
echo BBS    ${BBS} >> "${DIR}/run.config"
echo BSS    ${BSS} >> "${DIR}/run.config"
echo EXP    ${EXP} >> "${DIR}/run.config"
echo RNN    ${RNN} >> "${DIR}/run.config"
echo POL    ${POL} >> "${DIR}/run.config"
echo DAT    ${DAT} >> "${DIR}/run.config"
echo RUN    ${RUN} >> "${DIR}/run.config"

cp run-vracer-irl.py $RUNPATH
cp settings_irl.sh $RUNPATH
cp -r _model/ $RUNPATH
cp observations*${ENV}* $RUNPATH

pushd .
cd ${RUNPATH}

python3 run-vracer-irl.py --env ${ENV} --ebru ${EBRU} --dbs ${DBS} --bbs ${BBS} --bss ${BSS} --exp ${EXP} --rnn ${RNN} --pol ${POL} --run ${RUN} 

popd

python3 -m korali.plot --dir $DIR --out "${ENV}_${RUN}.png"
python3 -m korali.rlview --dir $DIR --out "irl-${ENV}_${RUN}.png" --showObservations --showCI 0.8
python3 -m korali.rlview --dir $DIR --out "firl-${ENV}_${RUN}.png" --featureReward --showObservations --showCI 0.8

date

code=$?

exit $code
