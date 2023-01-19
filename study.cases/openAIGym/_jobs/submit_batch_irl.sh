run=6000

export ENV="HalfCheetah-v4"
export POL="Linear"
export EXP=3000000
export DAT=100
export EBRU=5000

for D in 1 4 16 64
do 
    for B in 1 4 16 64
    do 
        run=$(($run+1))
        export RUN=$run
        export DBS=$D
        export BBS=$B
        bsub < bsub-vracer-irl.lsf
    done
done
