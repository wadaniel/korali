#!/bin/bash -l
#SBATCH --job-name="run-vracer-coord-test"
#SBATCH --output=run-vracer-coord_%j.out
#SBATCH --account=s929
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu

pushd ..

cat run-vracer-coord-v2.py

expDir=$SCRATCH/Cartpolev2_v0/$SLURM_JOB_ID
mkdir -p $expDir
cp run-vracer-coord-v2.py $expDir
cp model.py $expDir
cp -r _model_cartpole $expDir
cp -r Results $expDir
cp -r Utils $expDir
cp -r Visualization $expDir

popd

pushd $expDir

srun -n 6 python3 run-vracer-coord-v2.py --epoch 100 --batch 4 --lr 1e-2 --ws 4.0 --hid 10 --iniRetrain 2000 --retrain 5000 --trRewTh 500 --tarAvRew 470 --launchNum 2 --dumpBestTrajectory --conf 0.9700 --m "TestFinalv2Net10Ini2000Re5000_u0.001_Results_0.9700_1/" --expBetPolUp 0.001 --testFreq 1

popd

date
