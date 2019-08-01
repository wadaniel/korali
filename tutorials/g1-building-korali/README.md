# G.1 - Building a Korali Application

Korali provides a range of optimization and uncertainty quantification tools. To use these tools, a user needs to create a *Korali Application*. This application serves as a nexus between the computational models (user-provided C++ objects, Python modules, or pre-compiled applications), and the Korali's main solver engine.

To build a Korali application in Python, you simply need to include the corresponding korali module and declare an instance of the Korali engine:

```python
#!/usr/bin/env python3
import korali
k = korali.initialize()
...
```

## Creating a Korali Application

A Korali application encapsulates the configuration necessary to run the Korali engine. In this section, we describe the steps required to create a Korali Application:

### Step 1) Define Problem

The first step of building a Korali application is to indicate which problem to be solved. For example: 

  ```python
   k["Problem"]["Type"] = "Sampling";
  ```

For a complete list of Korali-supported problems, **[click here](/usage/problems.md)**.

### Step 2) Define Solver

The Solver module generates samples for evaluation, performs the statistical analysis, and outputs the results for the problem described in the Korali application. For example: 

  ```python
   k["Solver"]["Type"] = "TMCMC";
  ```

For a complete list of Korali-supported solvers, **[click here](/usage/solvers.md)**.

###Step 3) Define Variables

A Korali Problem describes the sampling space of the physical or real-world phenomenon to analyze. The number of variables defined by the user represents a dimension of the problem. Variables are created by simply adding their name to the problem configuration:

```python
# Example: Defining two variables
k["Variables"][0]["Name"] = "Thermal Conductivity"
k["Variables"][1]["Name"] = "Heat Source Position"
```

Variable definitions require additional parameters depending on which problem and solver types have been selected. The syntax for specifying these parameters is explained in each solver/problem page.

### Step 4) Define Conduit

The evaluation conduit module executes the computational model(s) for a given sample and returns their raw results back to the solver. The choice of conduit depends on the design and requirements of the computational model. For example:

	```python
	 k["Conduit"] = "External";
	```
For a complete list of Korali-supported conduits, **[click here](/usage/conduits.md)**.

### Step 5) Define Model(s)

During execution, the Korali engine will evaluate many *samples*. A Korali sample is a vector containing a particular set of values for all (computational and statistical) variables defined in the problem. Korali will stochastically produce new samples to be evaluated by a user-provided computational model.

For a complete list of Korali-supported models and their syntax, **[click here](/usage/models.md)**.

## Defining General Settings

The runtime Korali application can be easily configured by accessing the instance of the korali Engine. Here is a list of general settings that apply to any Korali app:

  ```python
      k["General"]["Verbosity"] = "Silent" # Do not produce console output.
  ```
  
For a complete list of Korali's general settings, **[click here](/usage/general.md)**.

## Running a Korali Application

A Python/C++ application can be run just like any other application, through a shell command:

```bash
> ./myApp myArguments
```

For distributed jobs, you can launch your Korali application with the SLURM/MPI/UPC++ launcher, specifying the number of cores, for example:

```bash
> srun -n 16 ./myApp myArguments
> mpirun -n 16 ./myApp myArguments
> upcxx-run -n 16 ./myApp myArguments
```

## Processing Results

Once the Korali engine has started running, it will execute non-preemptively until a termination criterion is met. Each solver method exposes a configurable set of termination criteria.

After execution, Korali will automatically create a results folder ```_korali_result``` where it will save the entire state of the engine (i.e., a *checkpoint*) after each generation. Users can perform several operations on these results:

- [Accessing Results](../results/#accessing-results)
- [Generate Plots](../results/#generate-plots)
- [Resume Execution](../results/#resume-execution)
