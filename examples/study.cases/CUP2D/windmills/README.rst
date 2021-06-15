Study Case: Flow-Control using Windmills
=========================================

Environment for windmills, described by the Windmill class (see _deps/CUP2D/Obstacles/Windmill{.cpp,.h}).

The **state** is given the orientation :math:`\theta`, and angular velocity :math:`\omega` of the fans.

The **action** is given by modifying the torque :math:`\tau`.

The **reward** is given by the energy :math:`r_1=\int \tau\omega\mathrm{d}t` and/or the difference between the flow-velocity and target-velocity :math:`r_2=\|\boldsymbol{u}-\boldsymbol{u}_0\|`.

Running / Evaluating the Reinforcement Learning
------------------------------------------------

The training and evaluation runs are runwith the following commands respectively:

.. code-block:: bash
	./batch-pair-windmill-vracer.sh output_folder
	./eval-pair-windmill-vracer.sh output_folder

To be specified:

* output_folder: folder of the output results of the training. Must be the same for training and evaluation calls.

Furthermore it contains the CUP2D specifications for the fluid simulations.



Details
-------

Source
^^^^^^

*_model/windmillEnvironment.cpp*: Defines the reinforcement learning agent characteristics, with the main RL simulation loop. 

To be specified:

line 123 && line 103
* Real en : energy factor
* Real flow : flow factor
* if action is set to zero (send 0 to act function), for the uncontrolled case


*run-vracer-windmill.cpp*: Defines Korali model with policy network and ReF-ER hyperparameters

To be specified:
line 78
* double max_torque : maximum torque applied to the fans


*eval-vracer-windmill.cpp*: RL evaluation program. 

To be specified:
* nothing

Study Cases
^^^^^^^^^^^

The simulations are launched with sbatch windmill.sbatch.

uncontrolled:
* set output_folder to uncontrolled
* set en to 0, flow to 0 and the value given to the act() functions to 0
* set max_torque to 1.0e-3

energy:
* set output_folder to energy_zero
* set en to 5.0e4, flow to 0 and the value given to the act() functions to action[i], where i is the number of the fan
* set max_torque to 1.0e-3

flow 1e-4:
* set output_folder to flow_zero
* set en to 0, flow to 2.5 and the value given to the act() functions to action[i], where i is the number of the fan
* set max_torque to 1.0e-4

both:
* set output_folder to both
* set en to 5.0e4, flow to 2.5 and the value given to the act() functions to action[i], where i is the number of the fan
* set max_torque to 1.0e-3

flow 1e-3:
* set output_folder to flow_zero
* set en to 0, flow to 2.5 and the value given to the act() functions to action[i], where i is the number of the fan
* set max_torque to 1.0e-3


Cleaning up results and plotting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to preprocess the results use

.. code-block:: bash
	python results.py

They can subsequently be plotted using

.. code-block:: bash
	python plot.py
