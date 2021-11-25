#!/bin/bash -l
#SBATCH --job-name="run-vracer-cartpole-v2"
#SBATCH --output=run-vracer-cartpole-v2_%j.out
#SBATCH --account=s929
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu

pushd ..

cat run-vracer-cartpole-v2.py

expDir=$SCRATCH/Surrogate_Cartpole_v2_VRACER
mkdir -p $expDir
cp run-vracer-cartpole-v2.py $expDir
cp -r _model_cartpole $expDir

popd

pushd $expDir

srun -n 6 python3 run-vracer-cartpole-v2.py --epoch 100 --batch 4 --lr 1e-2 --ws 4.0 --hid 10 --iniRetrain 500 --retrain 5000 --trRewTh 500 --tarAvRew 0 --launchNum 1 --dumpBestTrajectory --conf 1.0000 --m "TestFinalNet10Ini500Re5000_u1.0_Results_1.0000_1/" --expBetPolUp 1.0 --testFreq 1 --maxPolUp 50000 --maxGen 1000000

popd

date
