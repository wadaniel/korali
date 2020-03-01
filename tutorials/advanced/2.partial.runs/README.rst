Partial Runs
=====================================================

In this tutorial we show how to stop and restart a previous Korali run.
All scripts in this tutorial follow this structure:

Example Scripts
---------------------------
    + *run-cmaes.py*
    + *run-dea.py*
    + *run-executor.py*
    + *run-mcmc.py*
    + *run-multiple.py*
    + *run-rprop.py*
    + *run-tmcmc.py*

Run
---------------------------

Set a `Termination Criteria` and run:

.. code-block:: python

    print('------------------------------------------------------')
    print('Now running first 50 generations...')
    print('------------------------------------------------------')

    e["Solver"]["Termination Criteria"]["Max Generations"] = 50
    k.run(e)

Restart
---------------------------

Update `Termination Criteria` and restart with `run`:

.. code-block:: python

    print('------------------------------------------------------')
    print('Now running last 50 generations...')
    print('------------------------------------------------------')

    e["Solver"]["Termination Criteria"]["Max Generations"] = 100
    k.run(e)

