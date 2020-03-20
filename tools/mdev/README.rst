.. _mdev-tool:

***********************************
Korali Module Creation Tool
***********************************

Usage
========================

Syntax: :code:`python3 -m korali.mdev --name MODULE_NAME --type MODULE_TYPE`

Where:

  - :code:`--name` specifies the name of the new module to create. It can contain only letters, numbers, underscores and dashes.
  - :code:`--type` specifies the module type ('problem' or 'solver')

Examples
========================

Creating a new Problem
----------------------------------

In this example, we create a new problem module:

.. code-block:: bash
  
  > cd korali/
  > python3 -m korali.mdev --name myNewProblem --type problem
  
This command has created a new module in the folder - :code:`source/modules/problem/myNewProblem`:

 .. code-block:: bash
  
  > ls source/modules/problem/myNewProblem
  myNewProblem.config  myNewProblem._cpp  myNewProblem._hpp  README.rst

Creating a new Solver
-----------------------------------------

In this example, we create a new problem module:

.. code-block:: bash
  
  > cd korali/
  > python3 -m korali.mdev --name myNewSolver --type solver
  
This command has created a new module in the folder - :code:`source/modules/solver/myNewSolver`:

 .. code-block:: bash
  
  > ls source/modules/solver/myNewSolver
  myNewSolver.config  myNewSolver._cpp  myNewSolver._hpp  README.rst
  

