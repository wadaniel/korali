#for E in Ant-v2 HalfCheetah-v2 Hopper-v2 Humanoid-v2 HumanoidStandup-v2 Reacher-v2 Swimmer-v2 Walker-v2; 
for E in Ant-v2;
do 
    for D in "Normal" "Clipped Normal" "Squashed Normal"; 
    do 
        export ENV=$E; DIS=$D; 
        #./sbatch-vracer-openAI.sh test${E}_$D; 
        ./sbatch-vracer-openAI.sh OpenAI_csacle2; 
    done; 
done
