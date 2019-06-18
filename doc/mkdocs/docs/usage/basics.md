#Korali Basics

## Building a Korali Application

Korali provides a range of optimization and uncertainty quantification tools. To use these tools, a user needs to create a *Korali Application*. This application serves as a nexus between the computational models (user-provided C++ objects, Python modules, or pre-compiled applications), and the Korali's main solver engine, as illustrated below.

![](modules.png)

To build a Korali application in Python, you simply need to include the corresponding korali module and declare an instance of the Korali engine:

```python
#!/usr/bin/env python3
import korali
k = korali.Engine()
...
```

## Creating a Korali Application

A Korali application encapsulates the configuration necessary to run the Korali engine. In this section, we describe the steps required to create a Korali Application:

### Step 1) Define Problem

A Problem describes the objective function to be optimized/sampled from. We distinguish between two evaluation types: 

- [Direct](/usage/evaluation/direct)
- [Bayesian](/usage/evaluation/bayesian)

### Step 2) Define Solver

The Solver module generates new samples to be evaluated, performs the statistical analysis, and outputs the results for the problem described in the Korali application. Korali currently provides two types of solvers: *optimizers* and *samplers*:

Optimizers find the global maximum or minimum of the described problem. The detailed configuration of the currently implemented optimizers can be found below:

- [CMA-ES](/usage/solvers/optimizers/cmaes)
- [Constrained CMA-ES](/usage/solvers/optimizers/ccmaes)
	
Samplers map the probability distribution of samples in the variable space of the described problem. The detailed configuration of the currently implemented optimizers can be found below:

- [Markov-Chain Monte-Carlo](/usage/solvers/samplers/mcmc) 
- [Transitional Markov-Chain Monte-Carlo](/usage/solvers/samplers/tmcmc)
  
###Step 3) Define Variables

A Korali Problem describes the sampling space of the physical or real-world phenomenon to analyze. The number of variables defined by the user represents a dimension of the problem. Variables are created by simply adding their name to the problem configuration:

```python
# Defining two variables
k["Variables"][0]["Name"] = "Thermal Conductivity"
k["Variables"][1]["Name"] = "Heat Source Position"
```

For some, additional solver-specific information is required for each variable. For example, CMA-ES requires the definition of an initial mean and standard deviation:

```python
# Adding CMA-ES Specific Configuration
k["Variables"][0]["CMA-ES"]["Initial Mean"] = 10.0;
k["Variables"][0]["CMA-ES"]["Initial Standard Deviation"] = 1.0;
```

### Step 4) Define Conduit

The evaluation conduit module is in charge of providing an interface for the execution of computational models and evaluation of their results. A novelty in Korali is its support for *multi-intrusive* evaluation. That is, Korali can sample from computational models either *semi-instrusively*, through static linking or dynamic loading of Python modules; or *nonintrusively*, through shell command execution. Additionally, Korali supports the execution of distributed (MPI/UPC+++) applications through its *distributed* conduit. The choice of conduit depends on the design and requirements of the computational model. 

The rationale and configuration of the currently implemented conduits can be found below:

- [Semi-Intrusive](/usage/conduits/semi-intrusive)
- [Distributed](/usage/conduits/distributed)
- [Nonintrusive](/usage/conduits/nonintrusive)
	
  
You can use the following syntax to select the desired conduit:

```python
# Selecting the Nonintrusive Conduit
k["Conduit"]["Type"] = "Semi-Intrusive" # This is the default if conduit type is not specified
k["Conduit"]["Type"] = "Nonintrusive"
k["Conduit"]["Type"] = "Distributed"
```

### Step 5) Define Models

Computational Models

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

Once the Korali engine has started running, it will execute non-preemptively until a termination criterion is met. Each solver method exposes a configurable set of termination criteria.

After execution, Korali will automatically create a results folder ```_korali\_result``` where it will save the entire state of the engine (i.e., a *checkpoint*) after each generation. Users can perform several operations on these results:

- [Export to Format](/usage/results/export)
- [Resume Execution](/usage/results/resume)
- [Generate Plots](/usage/results/plotting)