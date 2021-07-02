Running Python MPI Applications
=====================================================

In this tutorial we show how a Python MPI model can be executed with Korali.

For more information on running Korali applications in parallel, see :ref:`Parallel Execution <parallel-execution>`. 
For more information on running Korali on MPI, see :ref:`Distributed Conduit <module-conduit-distributed>`. 

MPI Init
---------------------------

Do not forget to init MPI inside the Korali application:

.. code-block:: python

    from mpi4py import MPI

Distributed Conduit
---------------------------

Run with the `Distributed` conduit to benefit from parallelized model evaluations.
Note that we set `Ranks Per Worker` to assign a team of MPI processes to the model.

.. code-block:: cpp

    k.setMPIComm(MPI_COMM_WORLD);
    k["Conduit"]["Type"] = "Distributed";
    k["Conduit"]["Ranks Per Worker"] = workersPerTeam;
    k["Profiling"]["Detail"] = "Full";
    k["Profiling"]["Frequency"] = 0.5;

Run
---------------------------

Compile the script with the `Makefile`.
Run the script with an input argument (Ranks Per Worker):

.. code-block:: bash

    mpirun -n 13 ./run-tmcmc 4

