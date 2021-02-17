Study Case: Reinforcement Learning on Artificial Swimmers
==========================================================

In this study case, we replicate the fish swimming experiment from CSELab.

Dependencies
--------------------------

CubismUP-2D has the following prerequisite libraries:

- MPI, with the $MPICXX enviroment variable defined.
- HYPRE, with the $HYPRE_ROOT environment variable defined.
- GSL, with the $GSL_ROOT environment variable defined.
- HDF5, with the $HDF5_ROOT environment variable defined.
- FFTW, with the $FFTW_ROOT environment variable defined.

Setup
---------------------------

1) Install CubismUP-2D by running:

.. code-block:: bash

   ./install_cup.sh

2) Compile the CubismUP-2D RL-interface to Korali by running:

.. code-block:: bash
   
  make -j6

3) Run the learning algorithm:

.. code-block:: bash
   
  ./run.sh
