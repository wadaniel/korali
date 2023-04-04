run=10

#export ENV="HalfCheetah-v4"
export ENV="Swimmer-v4"
#export ENV="Hopper-v4"
#export ENV="Walker2d-v4"
#export POL="Quadratic"
export POL="Linear"
export EXP=5000000
export BSS=500
export DAT=100
export RNN=8

for EU in 1000 #5000
do
    for D in 4 16
    do 
        for B in 4 16 64
        do 
            run=$(($run+1))
            export RUN=$run
            export DBS=$D
            export BBS=$B
            export EBRU=$EU
            bsub < bsub-vracer-irl.lsf
        done
    done
done
