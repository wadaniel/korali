Study Case: Reinforcement Learning on Fish Swimming
=======================================================

In this study case, we replicate the fish swimming experiment from CSELab.

Dependencies
--------------------------

This study case has the following prerequisite libraries:

- HYPRE, with the $HYPRE_ROOT environment variable defined.
- GSL, with the $GSL_ROOT_DIR environment variable defined.
- HDF5, with the $HDF5_ROOT environment variable defined.

Setup
---------------------------

[Optional] Setup Cubism-related configuration through the CUBISM_BLOCK_SIZE and CUBISM_NTHREAD environment variables. For example:


.. code-block:: bash

   export CUBISM_BLOCK_SIZE=32
   export CUBISM_NTHREADS=8

1) Install CubismUP2D by running:

.. code-block:: bash

   ./install_deps.sh

2) Compile the study case by running:

.. code-block:: bash
   
  ./make -j6

3) Run the test:

.. code-block:: bash
   
  ./run_test.sh
