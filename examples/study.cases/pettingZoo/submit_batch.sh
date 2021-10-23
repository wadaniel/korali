# Launch continuous envs
for env in Multiwalker Waterworld
do 
    for model in 0 2
    do
        for run in 1 2 3 4 5 6 7 8 9 10 11
        do
            export ENV=$env
            export MODEL=$model
            export RUN=$run
            ./sbatch-vracer-zoo.sh
        done

    done
done

# Launch discrete envs
for env in Pursuit
do 
    for model in 0 2
    do
        for run in 1 2 3 4 5 6 7 8 9 10 11
        do
            export ENV=$env
            export MODEL=$model
            export RUN=$run
            ./sbatch-dvracer-zoo.sh 
        done
    done
done
