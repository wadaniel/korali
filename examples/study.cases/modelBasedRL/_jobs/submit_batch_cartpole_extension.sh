for retrain in 1000 2000 5000;
do
    for iniretrain in 500;
    do
        for conf in 0.8900 0.9100 0.9300 0.9500 0.9700 0.9950 0.9995;
        do
            export CONF=$conf
            export INI=$iniretrain
            export RE=$retrain
            export M="ThesisFinalNet10Ini${INI}Re${RE}_u1.0_Results_${CONF}_1/"
            ./sbatch-vracer-cartpole.sh
        done;
    done; 
done
