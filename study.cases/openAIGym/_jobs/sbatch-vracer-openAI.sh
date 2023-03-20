#!/bin/bash -l

source ../settings.sh

echo $ENV
echo $DIS
echo $L2
echo $OPT
echo $LR

cat > run.sh <<EOF
#!/bin/bash -l
#SBATCH --job-name="OpenAI_VRACER_$ENV"
#SBATCH --output=OpenAI_$ENV_%j.out
#SBATCH --error=OpenAI_$ENV_err_%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-core=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=s1160

RUNPATH=$SCRATCH/OpenAI/$ENV/\$SLURM_JOB_ID
mkdir -p \$RUNPATH

pushd ..

cat run-vracer.py

cp run-vracer.py \$RUNPATH
cp settings.sh \$RUNPATH
cp -r _model/ \$RUNPATH

popd

pushd \$RUNPATH

OMP_NUM_THREADS=12 python3 run-vracer.py --env "$ENV" --dis "$DIS" --l2 $L2 --opt $OPT --lr $LR

resdir=\$(ls -d _result_vracer_*)
DISNW="\$(echo -e "${DIS}" | tr -d '[:space:]')"
figureName1=vracer_irl_${ENV}.png
python3 -m korali.rlview --dir \$resdir --output \$figureName1
python3 -m korali.rlview --dir \$resdir --output \$figureName1
figureName1=vracer_irl_${ENV}.png
python3 -m korali.plot --dir \$resdir --output \$figureName2

popd

date
EOF

sbatch run.sh
