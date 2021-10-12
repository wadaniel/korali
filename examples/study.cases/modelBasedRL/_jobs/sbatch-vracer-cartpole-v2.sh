#!/bin/bash -l

#source ../settings_cartpole_v2.sh

echo $CONF
echo $INI
echo $RE
echo $M

cat > run.sh <<EOF
#!/bin/bash -l
#SBATCH --job-name="run-vracer-cartpole-v2_$CONF_$INI_$RE"
#SBATCH --output=run-vracer-cartpole-v2_$CONF_$INI_$RE_%j.out
#SBATCH --error=run-vracer-cartpole-v2_$CONF_$INI_$RE_err_%j.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-core=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=s929

RUNPATH=$SCRATCH/Surrogate_Cartpole_v2_VRACER
mkdir -p \$RUNPATH

pushd ..

cat run-vracer-cartpole-v2.py

cp run-vracer-cartpole-v2.py \$RUNPATH
#cp settings_cartpole_v2.sh \$RUNPATH
cp -r _model_cartpole/ \$RUNPATH

popd

pushd \$RUNPATH

srun -n 6 python3 run-vracer-cartpole-v2.py --epoch 100 --batch 4 --lr 1e-2 --ws 4.0 --hid 10 --iniRetrain $INI --retrain $RE --trRewTh 500 --tarAvRew 0 --launchNum 10 --maxGen 1000000 --dumpBestTrajectory --conf $CONF --m "${M}" --expBetPolUp 1.0 --testFreq 1 --maxPolUp 50000

popd

date
EOF

sbatch run.sh
