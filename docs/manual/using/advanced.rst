.. _advanced-features:

****************************
Advanced Features
****************************

Running a Korali Application
=================================================

Using Command-line Python
-----------------------------------------

Using Anaconda, Spyder, or iPython
------------------------------------------

Running a C++ Python-based Korali Application
----------------------------------------------

Running Multiple Experiments simultaneously
---------------------------------------------

Accessing Results
================================================

Configuring Results Output
-----------------------------------------------

Accessing results from a save-state file
-----------------------------------------------

Plotting Results
-----------------------------------------------


Parallel / Distributed Execution
================================================

Execution Conduit
-------------------------

The evaluation conduit module executes the computational model(s) for each sample and returns their raw results back to the solver. 

The choice of conduit depends on the design and requirements of the computational model. The simple (default) conduit is one that evaluates samples sequentially:

.. code-block:: python

  k["Conduit"]["Type"] = "Simple";

Below is a complete list of conduits currently implemented in Korali:

[**Execution Conduit List**](conduit)


Using the Concurrent Conduit
--------------------------------------------

Using the Distributed Conduit
--------------------------------------------

Running MPI-based Applications
-------------------------------------------

Running Multiple Experiments in Parallel
--------------------------------------------

Obtaining Profiling Information
--------------------------------------------


Resuming Jobs
=================================================

Running an experiment partially
------------------------------------------------

Resuming an experiment from a save-state file
------------------------------------------------

Resuming multiple experiments simultaneously
------------------------------------------------

