Study Case: Nek5000
=======================================================

This study case provides an environment that connects to Nek5000

Dependencies
--------------------------

This study case has the following prerequisites:

- Fortran 77 compiler, defined by the environment variable $F77 
- C compiler, defined by the environment variable $CC

Setup
---------------------------

0) [Optional] Change the experiments configuration set in the _config folder.

1) Install and configure Nek5000 

.. code-block:: bash

   ./install_deps.sh

2) Compile the study case by running:

.. code-block:: bash
   
  make

3) Run the agent:

.. code-block:: bash
   
  ./run-korali
