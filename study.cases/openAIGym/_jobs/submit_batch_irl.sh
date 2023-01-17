run=400

export ENV="HalfCheetah-v4"
export POL="Linear"
export EXP=5000000
export DAT=1000

for D in 4 16 64
do 
    for B in 4 16 64
    do 
        run=$(($run+1))
        export RUN=$run
        export DBS=$D
        export BBS=$B
        bsub < bsub-vracer-irl.lsf
        exit
    done
done
