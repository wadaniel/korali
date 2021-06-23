*******************************
Conduits
*******************************

Conduit modules specify how the computational models used to evaluate samples are to be executed, in particular regarding parallelism/concurrency. To select a specific conduit, use the following syntax:
 
.. code-block:: python

   k = korali.Engine()
   k["Conduit"]["Type"] = "Distributed"  
   
For more information, see `Parallel Execution <parallel-execution>`_. 
