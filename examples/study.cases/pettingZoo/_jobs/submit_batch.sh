# Launch continuous envs
for env in Multiwalker Waterworld
do 
    for model in 0 1
    do
        export ENV=$env
        export MODEL=$model
        ./sbatch-vracer-zoo.sh 
    done
done

# Launch discrete envs
for env in Pursuit
do 
    for model in 0 1
    do
        export ENV=$env
        export MODEL=$model
        ./sbatch-dvracer-zoo.sh 
    done
done
