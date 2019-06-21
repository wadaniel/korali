#Korali Basics

## Building a Korali Application

Korali provides a range of optimization and uncertainty quantification tools. To use these tools, a user needs to create a *Korali Application*. This application serves as a nexus between the computational models (user-provided C++ objects, Python modules, or pre-compiled applications), and the Korali's main solver engine, as illustrated below.

![](modules.png)

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

A Problem describes the objective function to be optimized/sampled from. We currently support these problem types: 

- [Direct Evaluation](../problems/direct)
- [Bayesian - Direct Likelihood](../problems/bayesian/direct)
- [Bayesian - Reference Likelihood](../problems/bayesian/reference)

### Step 2) Define Solver

The Solver module generates samples for evaluation, performs the statistical analysis, and outputs the results for the problem described in the Korali application. Korali currently provides two types of solvers: *optimizers* and *samplers*:

**Optimizers** find the global maximum or minimum of the described problem. The detailed configuration of the currently implemented optimizers can be found below:

- [CMA-ES](../solvers/optimizers/cmaes)
- [Constrained CMA-ES](../solvers/optimizers/ccmaes)
- [Differential Evolution](../solvers/optimizers/diffevo)
	
**Samplers** map the probability distribution of samples in the variable space of the described problem. The detailed configuration of the currently implemented optimizers can be found below:

- [Markov-Chain Monte-Carlo](../solvers/samplers/mcmc) 
- [Transitional Markov-Chain Monte-Carlo](../solvers/samplers/tmcmc)
  
###Step 3) Define Variables

A Korali Problem describes the sampling space of the physical or real-world phenomenon to analyze. The number of variables defined by the user represents a dimension of the problem. Variables are created by simply adding their name to the problem configuration:

```python
# Example: Defining two variables
k["Variables"][0]["Name"] = "Thermal Conductivity"
k["Variables"][1]["Name"] = "Heat Source Position"
```

Variable definitions require additional parameters depending on which problem and solver types have been selected. The syntax for specifying these parameters is explained in each solver/problem page. 

### Step 4) Define Conduit

The evaluation conduit module executes the computational model(s) for a given sample and returns their raw results back to the solver. The choice of conduit depends on the design and requirements of the computational model. 

The rationale and configuration of the currently implemented conduits can be found below:

- [Semi-Intrusive](../conduits/semi-intrusive)
- [Distributed](../conduits/distributed)
- [Nonintrusive](../conduits/nonintrusive)

### Step 5) Define Model(s)

During execution, the Korali engine will evaluate many *samples*. A Korali sample is a vector containing a particular set of values for all (computational and statistical) variables defined in the problem. Korali will stochastically produce new samples to be evaluated by a user-provided computational model.

Korali provides several ways to specify external code/applications to provide results to the Korali engine:

- [Simple](../models/simple)
- [Rerefence](../conduits/reference)
- [Constraint](../conduits/constraint)

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

- [Generate Plots](../results/#generate-plots)
- [Export Results](../results/#export-results)
- [Resume Execution](../results/#resume-execution)
		 