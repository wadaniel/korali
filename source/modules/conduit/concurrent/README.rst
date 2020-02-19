*******************************
Concurrent Conduit
*******************************

Many computational models are comprised of legacy codes that can be too complex or poorly-documented for non-expert users to modify. Korali's External conduit allows running any pre-compiled or legacy computational model. It requires that the user specifies the way in which variable values are passed to the model and how to parse the results from file or standard output. 

Korali's External conduit uses a fork/join strategy to instantiate an independent operating-system process per each sample. Since I/O and process management operations incur much more overhead than direct function calls, the External conduit may be less efficient than the Linked conduit.

**Usage**

*Launching an External Application*

The External conduit guarantees that each model is executed as an independent process. As such, the model can be suspended in wait for a sub-process to run, as shown in the following example:

.. code-block:: python

  def myModel(x):
     argString = ['./model/ackleyFunction.py']
     for i in range(x.getVariableCount()):
       argString.append(str(x.getVariable(i)))
     retValue = subprocess.check_output(argString)
     x.addResult(float(retValue.decode()))

The model in the example launches a new process that solves the Ackley function and outputs the result to Stdout. The model captures the output and returns it to Korali. Note that, although the model is suspended until the subprocess finishes, Korali continues evaluating samples normally.

The Korali application is then programmed as usual:

.. code-block:: python

  #!/usr/bin/env python3
  import korali
  k = korali.Engine()

  # Set problem, solver, and variables.

  # Set model.
  k.setModel(myModel)

  # Defining the External conduit
  k["Conduit"] = "External"

  # Setting concurrent jobs count. 
  # If this is larger than 1, then many samples will be evaluated in parallel
  # as separate processes. In this case, execution is limited to a single node.
  k["Conduit"]["Concurrent Jobs"] = 16

  k.run()

Then, we run our application normally:

.. code-block:: bash

   > ./myKoraliApp # Will run 16 processes!


*Launching an MPI Application*

In the case of MPI processes, the procedure is similar, but Korali needs to be launched within the scope of a job (e.g., SLURM). An example model is shown below:

.. code-block:: python

   def myModel(x):
     argString = ['mpirun -n 16 ./model/ackleyMPIFunction.py']
     for i in range(x.getVariableCount()):
       argString.append(str(x.getVariable(i)))
     retValue = subprocess.check_output(argString)
     x.addResult(float(retValue.decode()))

