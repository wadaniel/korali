Model Optimization: Finding the Global Maximum
===============================================

In this tutorial we show how to **optimize** a given function. 

Example Scripts
---------------

    + *run-cmaes.py*: Runs this tutorial using the CMAES solver.
    + *run-dea.py*: Runs this tutorial using the DEA solver.
    + *run-lmcma.py*: Runs this tutorial using the LMCMA solver.
    + *run-rprop.py*: Runs this tutorial using the Rprop solver.
    
Problem Description
------------------- 

We are given the function $f(\vartheta)=-\vartheta^2$ for $\vartheta\in[-10,10]$.
We want to find the maximum of the function in the given interval.

The Objective Function
----------------------

Create a folder named `model`. Inside, create a file with name `directModel.py` and paste the following code,
::
    #!/usr/bin/env python

    def evaluateModel(p):
        x = p["Parameters"][0]
        p["Evaluation"] = -x*x

This is the computational model that represents our objective function.


Optimization with CMAES
-----------------------

First, open a file (you could name it 'a1-optimization') and import the korali module
::
    #!/usr/bin/env python3
    import korali

Import the computational model,
::
    import sys
    sys.path.append('./model')
    from directModel import *

The Korali Engine and Experiment Objects
----------------------------------------

Next we construct a `korali.Engine` and a `korali.Experiment` object and set the computational model,
::
    k = korali.Engine()
    e = korali.Experiment()

    e["Problem"]["Objective Function"] = evaluateModel


The Problem Type
----------------

Then, we set the type of the problem to `Direct Evaluation`, and the objective to maximization,
::
    e["Problem"]["Type"] = "Evaluation/Direct/Basic"
    e["Problem"]["Objective"] = "Maximize"

A list of implemented solver- and problem types, although not optimally
reader friendly, can be found in [module.cpp](../../source/module.cpp).  

The Variables
-------------

In this problem there is only one variable, `X`, whose domain we set to [-10,10],
::
    e["Variables"][0]["Name"] = "X"
    e["Variables"][0]["Lower Bound"] = -10.0
    e["Variables"][0]["Upper Bound"] = +10.0

The Solver
----------
We choose the solver `CMAES`, set the population size to be `32` and two termination criteria,
::
    e["Solver"]["Type"] = "CMAES"
    e["Solver"]["Population Size"] = 32
    e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-7
    e["Solver"]["Termination Criteria"]["Max Generations"] = 100

For a detailed description of CMAES settings see [here](../../usage/solvers/cmaes.md).

Finally, we need to add a call to the run() routine to start the Korali engine.
::
    k.run(e)

Running
-------

We are now ready to run our example:
`./a1-optimization`

Or, alternatively:
`python3 ./a1-optimization`

The results are saved in the folder `_korali_result/`.

Plotting
--------
You can see the results of CMA-ES by running the command,
`python3 -m korali.plotter`
