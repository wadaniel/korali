run=300

export ENV="Swimmer-v4"
export POL="Quadratic"
export EXP=3000000

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
