for layers in 5;
do
    for units in 300;
    do
        for conf in 0.8000 0.8500 0.9000 0.9250 0.9500 0.9750 0.9900;
        do
	    for launch in 1 2 3 4 5;
	    do
	        for iniretrain in 1250000;
		do
	            for retrain in 5000000;
		    do
                
			export CONF=$conf
                        export INI=$iniretrain
                        export RE=$retrain
                        export LAYERS=$layers
			export UNITS=$units
			export LAUNCH=$launch
			export M="ThesisFinalWithTrickNet${LAYERS}x${UNITS}Ini${INI}Re${RE}_u1.0_Results_${CONF}_${LAUNCH}/"
                        export ENV=Swimmer-v2
                        export DIS="Clipped Normal"
			./sbatch-vracer-openAI-final-trick.sh
		    done;
		done;
	    done;
        done;
    done; 
done
