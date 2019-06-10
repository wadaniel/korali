# Configuring Korali

A Korali application encapsulates all the configuration necessary to run the Korali engine. In this section, we describe each of its main aspects:

+ [Variable Space](#variable-space)
+ [Evaluation Type](#evaluation-type)
+ [Solver Method](#solver-method)
+ [Evaluation Conduit](#evaluation-conduit)
+ [Computational Model](#computational-model)

## Variable Space

A Korali Problem describes the sampling space of the physical or real-world phenomenon to analyze. The number of variables defined by the user represents a dimension of the problem. Variables are configured by accessing the problem's variable array, specifying their constant or variable numerical index:

```python
# Accessing variable 0
k["Problem"]["Variables"][0] 
	
# Accessing variable i
k["Problem"]["Variables"][i]
```

Korali allows defining information about each variable, such as its name and prior probability distribution, for example:

```python
k["Problem"]["Variables"][0]["Name"] = "Thermal Conductivity"
k["Problem"]["Variables"][0]["Distribution"] = "Uniform"
```

Each prior distribution defines its own set of configurable parameters. The full list of prior distribution types and their configuration can be found at:

- [Cauchy](/usage/distributions/cauchy)
- [Exponential](/usage/distributions/exponential)
- [Gamma](/usage/distributions/gamma)
- [Gaussian](/usage/distributions/gaussian)
- [Laplace](/usage/distributions/laplace)
- [Uniform](/usage/distributions/uniform)
	

We make a distinction between computational and statistical variable types. Computational variables describe the dimension of the problem-space (e.g., the X, Y, and Z coordinates of a real-world 3D problem), while statistical variables are employed to infer values from the statistical model (e.g., the error estimation $\sigma$ of a Gaussian process).

```python
k["Problem"]["Variables"][0]["Type] = "Computational"
k["Problem"]["Variables"][1]["Type] = "Statistical"
```

The number of statistical variables required depends on the evaluation type selected, as described next.
	
### Evaluation Type

During execution, the Korali engine will evaluate many *samples*. A sample represents a vector containing a particular set of values for all (computational and statistical) variables defined in the problem. Korali's engine will stochastically produce new samples to be evaluated, depending on the solver method chosen. The evaluation returns a single value, called *fitness*, per evaluated sample. The value of the sample fitness depends on the evaluation type.

The evaluation type describes the objective function to be optimized/sampled from. We distinguish between two evaluation types: 

- [Direct](/usage/evaluation/direct)
- [Bayesian](/usage/evaluation/bayesian)

##Solver Method

The Solver module generates new samples to be evaluated, performs the statistical analysis, and outputs the results for the problem described in the Korali application. Korali currently provides two types of solvers: *optimizers* and *samplers*:

### Optimizers
 Optimizer solvers find the global maximum or minimum of the described problem. The detailed configuration of the currently implemented optimizers can be found below:

- [CMA-ES](/usage/solvers/optimizers/cmaes)
- [Constrained CMA-ES](/usage/solvers/optimizers/ccmaes)
	
You can use the following syntax to select, for example, the CMA-ES optimizer:

```python
# Selecting the CMA-ES optimizer method.
k["Solver"]["Method"] = "CMA-ES"
```

### Samplers

Sampler solver type maps the probability distribution of samples in the variable space of the described problem. The detailed configuration of the currently implemented optimizers can be found below:
 
- [Transitional Markov-Chain Monte-Carlo](/usage/solvers/samplers/tmcmc)
  
You can use the following syntax to select, for example, the TMCMC sampler:

```python
# Selecting the CMA-ES optimizer method.
k["Solver"]["Method"] = "TMCMC"
```

##Evaluation Conduit

The evaluation conduit module is in charge of providing an interface for the execution of computational models and evaluation of their results. A novelty in Korali is its support for *multi-intrusive* evaluation. That is, Korali can sample from computational models either *semi-instrusively*, through static linking or dynamic loading of Python modules; or *nonintrusively*, through shell command execution. Additionally, Korali supports the execution of distributed (MPI/UPC+++) applications through its *distributed* conduit. The choice of conduit depends on the design and requirements of the computational model. 

The rationale and configuration of the currently implemented conduits can be found below:

- [Semi-Intrusive](/usage/conduits/semi-intrusive)
- [Distributed](/usage/conduits/distributed)
- [Nonintrusive](/usage/conduits/nonintrusive)
	
  
You can use the following syntax to select, for example, the TMCMC sampler:

```python
# Selecting the Nonintrusive Conduit
k["Conduit"]["Type"] = "Nonintrusive"
```

##Computational Model

```python
# Defining Computational Model
k.setModel(myModel)
```