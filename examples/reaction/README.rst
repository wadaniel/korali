Reaction: Simulating the evolution of reactants
==========================================

In this tutorial we show how to **simulate** reactions using a stochastic simulator method (SSM).

The Problem Type
----------------

We set the type of the problem to `Reaction`

.. code-block:: python

    e["Problem"]["Type"] = "Reaction"


Problem Description
------------------- 

A set of reaction equations can be defined as follows:

.. code-block:: python

    e["Problem"]["Reactions"][0]["Equation"] = "S+I->2I"
    e["Problem"]["Reactions"][0]["Rate"] = 0.0005
    e["Problem"]["Reactions"][1]["Equation"] = "I->R"
    e["Problem"]["Reactions"][1]["Rate"] = 0.2


The Variables
-------------

Each reactant must be declared as a variable with an initial number of reactants

.. code-block:: python

    e["Variables"][0]["Name"] = "S"
    e["Variables"][0]["Initial Reactant Number"] = 5000

    e["Variables"][1]["Name"] = "I"
    e["Variables"][1]["Initial Reactant Number"] = 5

    e["Variables"][2]["Name"] = "R"
    e["Variables"][2]["Initial Reactant Number"] = 0


The Solver
----------

The solver method, the simulation length, the number of simulated trajectories and solver specific configurations are defined in the solver section of the korali application

.. code-block:: python

    e["Solver"]["Type"] = "SSM/SSA"
    e["Solver"]["Simulation Length"] = 20.
    e["Solver"]["Simulations Per Generation"] = 100
    e["Solver"]["Termination Criteria"]["Max Num Simulations"] = 1000
    e["Solver"]["Diagnostics"]["Num Bins"] = 500


Plotting
--------

You can see the averaged trajectories of the SSM by running the command `python3 -m korali.plot --dir _korali_result_sir_ssa`
