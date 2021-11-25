#!/bin/bash -l
#SBATCH --job-name="surr_openAI_VRACER_Swimmer-v2"
#SBATCH --output=surr_OpenAI_%j.out
#SBATCH --error=surr_OpenAI_%j.out
#SBATCH --time=15:00:00
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-core=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=s929

RUNPATH=/scratch/snx3000/jclapesr/Surrogate_OpenAIGym_VRACER/Swimmer-v2
mkdir -p $RUNPATH

pushd ..

cat run-vracer-openAIgym.py

cp run-vracer-openAIgym.py $RUNPATH
cp settings_openAIgym.sh $RUNPATH
cp -r _model_openAIgym/ $RUNPATH

popd

pushd $RUNPATH

OMP_NUM_THREADS=12 srun python3 run-vracer-openAIgym.py --env Swimmer-v2 --dis "Clipped Normal" --l2 0.0 --opt 0.1 --lrRL 0.0001 --lr 0.0005 --batch 512 --epoch 250 --epoch2 50 --layers 5 --units 300 --p 0.0 --ws 0.75 --conf 0.9900 --m "ThesisFinalWithTrickNet5x300Ini1250000Re5000000_u1.0_Results_0.9900_5/" --maxExp 5000000 --testFreq 10 --dumpBestTrajectory --iniRetrain 1250000 --retrain 5000000 --expBetPolUp 1 --launchNum 1 --start 5

resdir=$(ls -d _result_vracer_*)
python3 -m korali.rlview --dir $resdir --output vracer.png

popd

date
