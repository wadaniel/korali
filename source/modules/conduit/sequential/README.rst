*******************************
Sequential Conduit
*******************************

# Conduits / Linked

## Description

The Linked conduit exposes a simple interface for the execution of Python or C++ functions. This conduit requires users to develop an interface to the computational model that allows Korali to gather results directly from memory during runtime. 

The Linked conduit provides an efficient interface for passing samples and read results directly from the computational model's host memory without the intervention of the operating system for process creation or I/O operations. Furthermore, the simplicity of Linked conduit makes it ideal for accessible or open-source computational models for which a Korali-compatible interface can be easily developed.

### MPI Interface

The distributed conduit provides support for MPI distributed computational models. It works by defining a set of evaluation teams (see Figure). Evaluation teams comprise multiple cores (i.e.,MPI or UPC++ ranks) each which receivenew samples to be evaluated and return their results asynchronously. The conduit will distribute newsamples to teams as soon as they become free. By distributing sample arguments to cores speculatively, this conduit reduces system-wide load imbalance and communication overhead.

![](distributedConduit.png)

## Usage

### Sequential Execution

The Linked is the default option for a conduit in Korali. It executes a single instance of the computational model at a time, also returning its result to the solver one-by-one.

.. code-block:: python
   #!/usr/bin/env python3
   import korali
   k = korali.Engine()
   
   # Set problem, solver, variables, and model.
   ...
   
   # Defining the Linked conduit.
   # Not necessary since this is the default conduit
   k["Conduit"] = "Linked"
   
   k.run()

Then, we run our application normally:

.. code-block:: bash
   > ./myKoraliApp

### Parallel Korali / Simple Model

In this case, we want to execute *N* parallel instances of a sequential model, with only 1 rank per computational model evaluation. We define a simple Korali application:

.. code-block:: python
   #!/usr/bin/env python3
   import korali
   k = korali.Engine()
   
   # Set problem, solver, variables, and model.
   ...
   
   # Setting ranks per team to 1.
   # This is not really necessary since the default value is 1.
   k["Conduit"]["Ranks Per Team"] = 1
   
   k.run()

And then run the application in shell, using the corresponding job launcher. We need N+1 ranks since the Korali engine requires a rank to serve as master:

.. code-block:: bash
   > mpirun -n N+1 ./myKoraliApp

### Sequential Korali / Parallel Model

In this case, we want to execute a single instance of a sequential model at a time, with *M* ranks per computational model evaluation. We define another Korali application:

.. code-block:: python
   #!/usr/bin/env python3
   import korali
   k = korali.Engine()
   
   # Set problem, solver, variables.
   ...
   
   #Using a distributed (MPI) model
   def myModel(data): 
    # Grabbing a pointer to the team's MPI communicator
    x = data.getVariable(0)
    y = data.getVariable(1)
    comm = korali_obj.getCommPointer()
    fval = myMPIModel(comm, x0, ks)
    data.addResult(fval)
    
   k.setModel(myModel);
   
   # Setting ranks per team to M.
   k["Conduit"]["Ranks Per Team"] = M
   
   k.run()

And then run the application in shell, using the corresponding job launcher:

.. code-block:: bash
   > mpirun -n M+1 ./myKoraliApp

### Parallel Korali / Paralell Model

In this case, we want to execute N instance of a sequential model at a time, with *M* ranks per computational model evaluation. We define another Korali application:

.. code-block:: python
   #!/usr/bin/env python3
   import korali
   k = korali.Engine()
   
   # Set problem, solver, variables.
   ...
   
   #Using the same MPI model as above.
   k.setModel(myModel);
   
   # Setting ranks per team to M.
   k["Conduit"]["Ranks Per Team"] = M
   
   k.run()

And then run the application in shell, using the corresponding job launcher:

.. code-block:: bash
   > mpirun -n N*M+1 ./myKoraliApp
