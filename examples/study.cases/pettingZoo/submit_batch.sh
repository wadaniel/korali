# Launch continuous envs
for env in Multiwalker Waterworld
do 
    for model in {0..5}
    do
        for run in {0..9}
        do
            for multi in false true 
            do
                export ENV=$env
                export MODEL=$model
                export RUN=$run
                export MULTI=$multi 
                ./sbatch-vracer-zoo.sh
            done
        done
    done
done

# Launch discrete envs
for env in Pursuit
do 
    for model in {0..5}
    do
        for run in {0..9}
        do
            for multi in false true
            do
                export ENV=$env
                export MODEL=$model
                export RUN=$run
                export MULTI=$multi
                ./sbatch-dvracer-zoo.sh 
            done
        done
    done
done
