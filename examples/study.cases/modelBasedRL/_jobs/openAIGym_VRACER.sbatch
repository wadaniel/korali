#!/bin/bash -l
#SBATCH --job-name="surr_openAIGym_VRACER"
#SBATCH --output=surr_openAIGym_VRACER_%j.out
#SBATCH --time=00:20:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=s929
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu

# Choose Environment
#env=Ant-v2
#env=HalfCheetah-v2
#env=Hopper-v2
#env=Humanoid-v2
#env=HumanoidStandup-v2
#env=InvertedDoublePendulum-v2
#env=InvertedPendulum-v2
env=Reacher-v2
#env=Swimmer-v2
#env=Walker2d-v2

#env=AntBulletEnv-v0
#env=HalfCheetahBulletEnv-v0
#env=HopperBulletEnv-v0
#env=HumanoidBulletEnv-v0
#env=Walker2DBulletEnv-v0

# Choose Policy Distribution
#dis="Normal"
#dis="Squashed Normal"
dis="Clipped Normal"
#dis="Truncated Normal"

# l2 regularization
l2=0.0

# off policy target
opt=0.1

# learning rate
lrRL=0.0001

# surrogate and korali arguments
lr=0.01
batch=128
epoch=100
hid=(15 15)
ws=1
conf=0.95
m="TestNet$SLURM_JOB_ID"
expBetPolUp=1
iniRetrain=65536
retrain=50000
launchNum=1

pushd ..

cat run-vracer-openAIgym.py

#expDir=$SCRATCH/Surrogate_OpenAIGym_VRACER/$env/$SLURM_JOB_ID
expDir=$SCRATCH/Surrogate_OpenAIGym_VRACER/$env/Results/$m
mkdir -p $expDir
cp run-vracer-openAIgym.py $expDir
cp -r _model_openAIgym $expDir

popd

pushd $expDir

OMP_NUM_THREADS=12 srun python3 run-vracer-openAIgym.py --env $env --dis "$dis" --l2 $l2 --opt $opt --lrRL $lrRL --lr $lr --batch $batch --epoch $epoch --hid $hid --ws $ws --conf $conf --m "$m" --dumpBestTrajectory --iniRetrain $iniRetrain --retrain $retrain --expBetPolUp $expBetPolUp --launchNum $launchNum
resdir=$(ls -d _result_vracer_*)

python3 -m korali.rlview --dir $resdir --output vracer.png

popd

date
