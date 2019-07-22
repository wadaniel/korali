#Korali Basics

## Building a Korali Application

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

A Problem describes the objective function to be optimized/sampled from. We currently support these problem types:

- [Optimization](../problems/optimization)
- [Constrained Optimization](../problems/constrained)
- [Sampling](../problems/bayesian/sampling)
- [Bayesian Inference](../problems/bayesian)
- [Hierarchical Bayesian](../problems/hierarchical)

### Step 2) Define Solver

The Solver module generates samples for evaluation, performs the statistical analysis, and outputs the results for the problem described in the Korali application. Korali currently provides two types of solvers: *optimizers* and *samplers*:

??? abstract "CMA-ES"

    To select the CMA-ES solver, use the following syntax:

  ```python
   k["Solver"] = "CMA-ES";
  ```

  [Click here for detailed information on CMA-ES.](../solvers/cmaes)

??? abstract "Differential Evolution Algorithm"

    To select the Differential Evolution Algorithm solver, use the following syntax:

  ```python
   k["Solver"] = "DEA";
  ```

  [Click here for detailed information on Differential Evolution](../solvers/de)

??? abstract "Markov-Chain Monte-Carlo"

    To select the Markov-Chain Monte-Carlo solver, use the following syntax:

  ```python
   k["Solver"] = "MCMC";
  ```

  [Click here for detailed information on MCMC](../solvers/mcmc)

??? abstract "Transitional Markov-Chain Monte-Carlo"

    To select the Transitional Markov-Chain Monte-Carlo solver, use the following syntax:

  ```python
   k["Solver"] = "TMCMC";
  ```

  [Click here for detailed information on TMCMC.](../solvers/tmcmc)
  
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

??? abstract "Linked"

    To select the Linked conduit, use the following syntax:

	```python
	 # This is not really necessary since Linked is the default conduit.
	 k["Conduit"] = "Linked";
	```

	[Click here for detailed information on the Linked conduit.](../conduits/linked)

??? abstract "External"

    To select the External conduit, use the following syntax:

	```python
	 k["Conduit"] = "External";
	```

	[Click here for detailed information on the External conduit.](../conduits/external)


### Step 5) Define Model(s)

During execution, the Korali engine will evaluate many *samples*. A Korali sample is a vector containing a particular set of values for all (computational and statistical) variables defined in the problem. Korali will stochastically produce new samples to be evaluated by a user-provided computational model.

Korali provides several ways to specify external code/applications to provide results to the Korali engine:

??? abstract "Direct"

   	[Click here for detailed information on Direct models.](../models/direct)

??? abstract "Reference"

   	[Click here for detailed information on Reference models.](../models/reference)	

??? abstract "Constraint"

   	[Click here for detailed information on Constraint models.](../models/constraint)	

## Defining General Settings

The runtime Korali application can be easily configured by accessing the instance of the korali Engine. Here is a list of general settings that apply to any Korali app:

??? abstract "Verbosity"

  Determines the amount of information printed by Korali during rume:

  + Default Value: "Normal"
  + Syntax:

  ```python
      k["Verbosity"] = "Silent" # No output at all.
      k["Verbosity"] = "Minimal" # Minimal progress report and results.
      k["Verbosity"] = "Normal" # Balanced progress report and results.
      k["Verbosity"] = "Detailed" # Verbose progress report, runtime details, and results.
  ```
  
??? abstract "Seed"

  Determines the random seed to use for random number generation

  + Default Value: *taken from /dev/random/*
  + Syntax:

  ```python
      k["Seed"] = 0xC0FFEE # Example.
  ```
  
??? abstract "Result Directory"

  Determines the path to the directory where to save the results of a Korali run.

  + Default Value: *_korali_results*
  + Syntax:

  ```python
      k["Result Directory"] = './my_result_path'
  ```

??? abstract "Max Generations"

  Determines how many solver generations to run before stopping execution.
  Execution can be resumed at a later moment.

  + Default Value: *no limit*
  + Syntax:

  ```python
      k["Termination Criteria"]["Max Generations"] = 50
  ```

??? abstract "Max Function Evaluations"

  Determines how many computational model evaluations to run before stopping execution.
  Execution can be resumed at a later moment.

  + Default Value: *no limit*
  + Syntax:

  ```python
      k["Termination Criteria"]["Max Function Evaluations"] = 5000
  ```
  
??? abstract "Console Output Frequency"

  Determines the frequency in which Korali will output results to console.
  A value of "1" means Korali will output to console results at every generation.

  + Default Value: 1
  + Syntax:

  ```python
      k["Console Output Frequency"] = 5
  ```
  
??? abstract "File Output Frequency"

  Determines the frequency in which Korali will output results to the result directory.
  A value of "1" means Korali will output result files at every generation.

  + Default Value: 1
  + Syntax:

  ```python
      k["File Output Frequency"] = 5
  ```

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
