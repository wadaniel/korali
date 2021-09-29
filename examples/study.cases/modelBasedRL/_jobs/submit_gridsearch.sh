for E in Reacher-v2;
do
    for D in "Clipped Normal";
    do
        for layers in 2 3;
        do
            for units in 10 15 20;
            do
                for lr in 0.001 0.01;
                do
                    for batch in 64 128 256;
                    do
                        for epoch in 100;
                        do
                            for p in 0.05;
                            do
                                for size in 65536;
                                do
                                    for ws in 1 2 3 4;
                                    do


                                        export ENV=$E
                                        export DIS="$D"
                                        export LR=$lr
                                        export BATCH=$batch
                                        export EPOCH=$epoch
                                        export LAYERS=$layers
                                        export UNITS=$units
                                        export P=$p
                                        export SIZE=$size
                                        export WS=$ws
                                        ./sbatch-grid-search-openAI.sh
                                    done;
                                done;
                            done;
                        done;
                    done;
                done;
            done;
        done;
    done;
done
