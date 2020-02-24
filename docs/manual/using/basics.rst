***********************
Korali Basics
***********************

Korali provides a range of optimization and uncertainty quantification tools. To use these tools, a user needs to create a *Korali Application*. This application serves as a nexus between the computational models (user-provided C++ objects, Python modules, or pre-compiled applications), and the Korali's main solver engine. A Korali application is a Python or C++ script that contains one or more evaluation of Korali experiments, which represent a set of problem/solver/conduit to be evaluated. 


Defining a Korali Experiment
========================================

A user creates an experiment by instantiating an object of type Experiment(). To run, the user simply calls korali.run().

```python
# Example: Creating a new Experiment
e = korali.Experiment()
# Configuring Experiment
korali.run(e)
```

Alternatively, the user can create a list of experiments to run concurrently:

```python
# Example: Creating a set of experiments.
eList = []
for i in range(10)
 e = korali.Experiment()
 # Configuring Experiment
 eList.append(e)
korali.run(eList) 
```

Each experiment can be configured with different general settings, as shown in the link below:
After configuring the experiment, the user chooses and configures its modules as shown below.

Problem Type
------------------------------

A Korali Problem describes the sampling space of the physical or real-world phenomenon to analyze. The number of variables defined by the user represents a dimension of the problem. Variables are created by simply adding their name to the problem configuration:


Variables
------------------------------

```python
# Example: Defining two variables
k["Variables"][0]["Name"] = "Thermal Conductivity"
k["Variables"][1]["Name"] = "Heat Source Position"
```

Variable definitions require additional parameters depending on which problem and solver types have been selected. These parameters are explained in detail at each solver/problem page.

Then, we need to tell Korali what information we would like to know about the problem. We call this 'Problem Type'. 

For example, the following type will sample the probability distribution of the variables: 

  ```python
   k["Problem"]["Type"] = "Sampling";
  ```

Below is a complete list of Korali-supported problem types:

[**Problem Type List**](problem)

Solver Method
--------------------------

The next step is to choose which solver algorithm should be used to obtain the results required by the problem. We call this 'Solver Type'. 

For example, the following solver will obtain the probability distribution using the TMCMC method: 

  ```python
   k["Solver"]["Type"] = "TMCMC";
  ```

Below is a complete list of solvers currently implemented in Korali:

[**Solver Method List**](solver)

Execution Conduit
-------------------------

The evaluation conduit module executes the computational model(s) for each sample and returns their raw results back to the solver. 

The choice of conduit depends on the design and requirements of the computational model. The simple (default) conduit is one that evaluates samples sequentially:

 ```python
  k["Conduit"]["Type"] = "Simple";
 ```
Below is a complete list of conduits currently implemented in Korali:

[**Execution Conduit List**](conduit)

.. _korali-sample:

Korali Samples
-----------------------

Function(Sample)

Interface to the user's computational model(s)
====================================================


