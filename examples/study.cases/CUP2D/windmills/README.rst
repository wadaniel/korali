Study Case: Flow-Control using Windmills
=========================================


Running the code
----------------

1) Parameters of the code

- windmill.sbatch: sbatch file for the slurm job launcher. The header has the following format

	#!/bin/bash -l
	#SBATCH --job-name="job_name"
	#SBATCH --output=job_name%j.out
	#SBATCH --time=24:00:00
	#SBATCH --nodes=65
	#SBATCH --ntasks-per-core=1
	#SBATCH --ntasks-per-node=1
	#SBATCH --account=s929
	#SBATCH --cpus-per-task=1
	#SBATCH --partition=normal
	#SBATCH --constraint=gpu
	
It launches the training and evaluation runs with the following commands respectively:

* ./batch-pair-windmill-vracer.sh output_folder
* ./eval-pair-windmill-vracer.sh output_folder

To be specified:
* output_folder: folder of the output results of the training. Must be the same for training
and evaluation calls. 
* job_name: suitable job name


- batch-pair-windmill-vracer.sh && eval-pair-windmill-vracer.sh : shell files with the CUP2D
specifications for the fluid simulations. They have the following format and parameters

	#!/bin/bash
	# Set number of nodes here
	mpiflags="mpirun -n 2"

	if [ ! -z $SLURM_NNODES ]; then
	 N=$SLURM_NNODES
	 mpiflags="srun -N $N -n $(($N)) -c 1"
	fi

	set -x

	# Defaults for Options
	BPDX=${BPDX:-8}
	BPDY=${BPDY:-8}
	LEVELS=${LEVELS:-4}
	RTOL=${RTOL-0.1}
	CTOL=${CTOL-0.01}
	EXTENT=${EXTENT:-1.4}
	CFL=${CFL:-0.22}
	# Defaults for Objects
	XPOS=${XPOS:-0.2}

	YPOS1=${YPOS:-0.6}
	YPOS2=${YPOS2:-0.8}

	XVEL=${XVEL:-0.15}

	MAAXIS=${MAAXIS:-0.0405}
	MIAXIS=${MIAXIS:-0.0135}
	NU=${NU:-0.0001215}



	OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -tdump 0.1 -nu $NU -tend 0 -muteAll 0 -verbose 1"
	OBJECTS="windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS1 bForced=1 bFixed=1 xvel=$XVEL tAccel=0 bBlockAng=0
	windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS2 bForced=1 xvel=$XVEL tAccel=0 bBlockAng=0

with one of the two following options for the training, respectively the evaluation run.

* $mpiflags ./run-vracer-windmill ${OPTIONS} -shapes "${OBJECTS}" $1
* $mpiflags ./eval-vracer-windmill ${OPTIONS} -shapes "${OBJECTS}" $1

To be specified:

* nothing


- _model/windmillEnvironment.cpp: Defines the reinforcement learning agent characteristics, 
with the main RL simulation loop. 

To be specified:

line 123 && line 103
* Real en : energy factor
* Real flow : flow factor
* if action is set to zero (send 0 to act function), for the uncontrolled case


- run-vracer-windmill.cpp : Defines Korali model with policy network and ReF-ER hyperparameters

To be specified:
line 78
* double max_torque : maximum torque applied to the fans


- eval-vracer-windmill.cpp :

To be specified:
* nothing


2) Running the train/eval code

The simulations are launched with sbatch windmill.sbatch.

- uncontrolled:
* set output_folder to uncontrolled
* set en to 0, flow to 0 and the value given to the act() functions to 0
* set max_torque to 1.0e-3

- energy:
* set output_folder to energy_zero
* set en to 5.0e4, flow to 0 and the value given to the act() functions to action[i], where i is the number of the fan
* set max_torque to 1.0e-3

- flow 1e-4:
* set output_folder to flow_zero
* set en to 0, flow to 2.5 and the value given to the act() functions to action[i], where i is the number of the fan
* set max_torque to 1.0e-4

- both:
* set output_folder to both
* set en to 5.0e4, flow to 2.5 and the value given to the act() functions to action[i], where i is the number of the fan
* set max_torque to 1.0e-3

- flow 1e-3:
* set output_folder to flow_zero
* set en to 0, flow to 2.5 and the value given to the act() functions to action[i], where i is the number of the fan
* set max_torque to 1.0e-3


3) Cleaning up results and plotting:

Run python results.py
Run python plot.py

- 
