        for layers in 1 2 3;
        do
            for units in 10 20 50;
            do
                    for batch in 4 16 32;
                    do
                        for epoch in 100;
                        do
                            for p in 0.05;
                            do
                                for size in 500;
                                do
                                    for ws in 2.0 4.0;
                                    do


                                        export LRS=$lr
                                        export BATCH=$batch
                                        export EPOCH=$epoch
                                        export LAYERS=$layers
                                        export UNITS=$units
                                        export P=$p
                                        export SIZE=$size
                                        export WS=$ws
                                        ./sbatch-grid-search-cartpole.sh
                                    done;
                                done;
                            done;
                        done;
                    done;
                done;
            done;
