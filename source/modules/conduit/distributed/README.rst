*******************************
Distributed Conduit
*******************************

This distributed conduit uses MPI to distribute sample evaluation among *n* workers. Each worker consists of *k* MPI ranks, where *k* is a configurable parameter. Communication among workers is realized via MPI messages.

This model is ideal for when your computational model can be directly linked with Korali and/or expects an MPI communicator itself. (see: `Running MPI Example <feature_running.mpi>`_). 

For more information, see `Parallel Execution <parallel-execution>`_. 

