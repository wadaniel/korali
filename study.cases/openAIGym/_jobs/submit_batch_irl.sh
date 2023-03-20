run=1

#export ENV="HalfCheetah-v4"
#export ENV="Swimmer-v4"
export ENV="Hopper-v4"
#export ENV="Walker2d-v4"
#export POL="Quadratic"
export POL="Linear"
export EXP=5000000
export DAT=100

for EU in 5000
do
    for D in 4 16
    do 
        for B in 4 16 64
        do 
            for R in 8 32
            do
                run=$(($run+1))
                export RUN=$run
                export DBS=$D
                export BBS=$B
                export EBRU=$EU
                export RNN=$R
                bsub < bsub-vracer-irl.lsf
            done
        done
    done
done
