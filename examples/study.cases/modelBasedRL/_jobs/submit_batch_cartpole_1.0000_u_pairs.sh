for retrain in 2000;
do
    for iniretrain in 500;
    do
        for conf in 1.0000;
        do
            for u in 0.0099 0.0192 0.0101 0.0242 0.0421 0.0284 0.0511 0.0734 0.0833;
            do
	        export CONF=$conf
                export INI=$iniretrain
                export RE=$retrain
		export U=$u
                export M="RealThesisFinalNet10Ini${INI}Re${RE}_u${U}_Results_${CONF}_1/"
                ./sbatch-vracer-cartpole-1.0000-u.sh
            done;
	done;
    done; 
done
