Study Case: Reinforcement Learning on Artificial Bacterial Flagella (ABF) in 3D
=================================================================================

In this study case, we drive a magnetic field to guide ABF towards a common goal in a 3D space.

Setup
---------------------------

1) Install Microswimmers ODE solver

.. code-block:: bash

   ./install_deps.sh

2) Compile the study case by running:

.. code-block:: bash
   
  make -j6

3) Run the test:

.. code-block:: bash
   
  ./run_test.sh
