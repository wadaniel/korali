for retrain in 2000;
do
    for iniretrain in 500;
    do
        for conf in 1.0000;
        do
            for u in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5;
            do
	        export CONF=$conf
                export INI=$iniretrain
                export RE=$retrain
		export U=$u
                export M="ThesisFinalNet10Ini${INI}Re${RE}_u${U}_Results_${CONF}_1/"
                ./sbatch-vracer-cartpole-1.0000-u.sh
            done;
	done;
    done; 
done
