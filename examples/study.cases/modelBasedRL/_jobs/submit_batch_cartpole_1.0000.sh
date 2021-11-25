for retrain in 5000;
do
    for iniretrain in 500;
    do
        for conf in 1.0000;
        do
            export CONF=$conf
            export INI=$iniretrain
            export RE=$retrain
            export M="ThesisFinalNet10Ini${INI}Re${RE}_u1.0_Results_${CONF}_1/"
            ./sbatch-vracer-cartpole-1.0000.sh
        done;
    done; 
done
