#!/bin/bash -l

source ../settings_openAIgym.sh

echo $ENV
echo $DIS
echo $L2
echo $OPT
echo $LR
echo $LAYERS
echo $UNITS
echo $CONF
echo $M
echo $INI
echo $RE
echo $LAUNCH

cat > run.sh <<EOF
#!/bin/bash -l
#SBATCH --job-name="surr_openAI_VRACER_$ENV"
#SBATCH --output=surr_OpenAI_$ENV_%j.out
#SBATCH --error=surr_OpenAI_$ENV_err_%j.out
#SBATCH --time=15:00:00
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-core=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=s929

RUNPATH=$SCRATCH/Surrogate_OpenAIGym_VRACER/$ENV
mkdir -p \$RUNPATH

pushd ..

cat run-vracer-openAIgym.py

cp run-vracer-openAIgym.py \$RUNPATH
cp settings_openAIgym.sh \$RUNPATH
cp -r _model_openAIgym/ \$RUNPATH

popd

pushd \$RUNPATH

OMP_NUM_THREADS=12 srun python3 run-vracer-openAIgym.py --env $ENV --dis "${DIS}" --l2 $L2 --opt $OPT --lrRL $LR --lr 0.0005 --batch 512 --epoch 250 --epoch2 50 --layers $LAYERS --units $UNITS --p 0.0 --ws 0.75 --conf $CONF --m "${M}" --maxExp 5000000 --testFreq 10 --dumpBestTrajectory --iniRetrain $INI --retrain $RE --expBetPolUp 1 --launchNum 1 --start $LAUNCH

resdir=\$(ls -d _result_vracer_*)
python3 -m korali.rlview --dir \$resdir --output vracer.png

popd

date
EOF

sbatch run.sh
