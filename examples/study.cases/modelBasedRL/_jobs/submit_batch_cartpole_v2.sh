for conf in 0.8500 0.8750 0.9000 0.9100 0.9200 0.9300 0.9400 0.9500 0.9600 0.9700 0.9800 0.9900 0.9950 0.9990 0.9995;
do 
    for iniretrain in 500;
    do
        for retrain in 5000;
        do
            export CONF=$conf
            export INI=$iniretrain
            export RE=$retrain
            export M="Finalv2Net10Ini${INI}Re${RE}_u1.0_Results_${CONF}_1/"
            ./sbatch-vracer-cartpole-v2.sh
        done;
    done; 
done
