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

parser.add_argument("--lr", type=float, default=1e-2, help="Initial learning rate surrogate nets")
parser.add_argument("--batch", type=int, default=8, help="Batch size")
parser.add_argument("--epoch", type=int, default=100, help="Total number of epochs to train the surrogate nets")
parser.add_argument("--hid", nargs="+", type=int, help="Specify integers for list of hidden sizes, starting ith input size to first hidden layer and finishing with output size of last layer")
parser.add_argument("--ws", type=float, default=4.0, help="Weight parameter for GaussianNLLLoss of reward (scalar)")
parser.add_argument("--conf", type=float, default=0.9500, help="Confidence Hyperparameter")
parser.add_argument("--m", type=str, default="", help="Prefix results and directories")
parser.add_argument("--dumpBestTrajectory", action='store_true', help="Dump best trajectory")
parser.add_argument("--trRewTh", type=int, default=495, help="trainingRewardThreshold")
parser.add_argument("--tarAvRew", type=int, default=495, help="targetAverageReward")
parser.add_argument("--maxGen", type=int, default=1000, help="maximumGenerations")
parser.add_argument("--epPerGen", type=int, default=1, help="episodesPerGeneration")
parser.add_argument("--testFreq", type=int, default=100, help="testingFrequency")
parser.add_argument("--polTestEp", type=int, default=20, help="policyTestingEpisodes")
parser.add_argument("--expBetPolUp", type=float, default=1, help="Experiences Between Policy Updates")
parser.add_argument("--maxPolUp", type=float, default=0, help="Maximum Policy Updates")
parser.add_argument("--maxSize", type=float, default=262144, help="Maximum size of memory for experience replay")
parser.add_argument("--launchNum", type=int, default=1, help="Number of times to run vracer.")
parser.add_argument("--iniRetrain", type=int, default=65536, help="Initial retraining sample data points.")
parser.add_argument("--retrain", type=int, default=50000, help="Retraining sample data points.")
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
