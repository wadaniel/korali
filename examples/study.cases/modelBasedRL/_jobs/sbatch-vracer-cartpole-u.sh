#!/bin/bash -l

#source ../settings_cartpole.sh

echo $CONF
echo $INI
echo $RE
echo $M
echo $U

cat > run.sh <<EOF
#!/bin/bash -l
#SBATCH --job-name="run-vracer-cartpole_$CONF_$INI_$RE"
#SBATCH --output=run-vracer-cartpole_$CONF_$INI_$RE_%j.out
#SBATCH --error=run-vracer-cartpole_$CONF_$INI_$RE_err_%j.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-core=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=s929

RUNPATH=$SCRATCH/Surrogate_Final_Cartpole_VRACER
mkdir -p \$RUNPATH

pushd ..

cat run-vracer-cartpole.py

cp run-vracer-cartpole.py \$RUNPATH
#cp settings_cartpole.sh \$RUNPATH
cp -r _model_cartpole/ \$RUNPATH

popd

pushd \$RUNPATH

srun -n 6 python3 run-vracer-cartpole.py --epoch 100 --batch 4 --lr 1e-2 --ws 4.0 --hid 10 --iniRetrain $INI --retrain $RE --trRewTh 500 --tarAvRew 0 --launchNum 10 --maxGen 1000000 --dumpBestTrajectory --conf $CONF --m "${M}" --expBetPolUp $U --testFreq 1 --maxPolUp 100000

popd

date
EOF

sbatch run.sh
