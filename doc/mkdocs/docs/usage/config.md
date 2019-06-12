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
	
  
You can use the following syntax to select the desired conduit:

```python
# Selecting the Nonintrusive Conduit
k["Conduit"]["Type"] = "Semi-Intrusive" # This is the default if conduit type is not specified
k["Conduit"]["Type"] = "Nonintrusive"
k["Conduit"]["Type"] = "Distributed"
```

## Computational Model

During execution, the Korali engine will evaluate many *samples*. A Korali sample is a vector containing a particular set of values for all (computational and statistical) variables defined in the problem. Korali will stochastically produce new samples to be evaluated by a user-provided computational model.

### Interface

Korali exposes a single interface for computational models, regardless of the chosen conduit/evaluation type. The computational model is specified via Korali's ```setModel(myModel)``` function, which receieves a function pointer that accepts a single argument of type ```KoraliData```.

Example for C++:

```C++
// Using a traditional C++ function pointer
void myModel(KoraliData& data) { /* do stuff */ };
k.setModel(&myModel);

// Using a lambda expression
auto myModel = [](Korali::ModelData& data) { /* do stuff */ };
k.setModel(myModel);
```

Example for Python:
```python
# Using a normal function definition
def myModel(data): #do stuff
k.setModel(myModel);

# Using a lambda expression
myModel = lambda koraliData: /* do stuff */ 
k.setModel(myModel);
```

### Reading Parameters 
The ```KoraliData``` object exposes information about the sample's computational variables. The snippet below shows the relevant operations:

```python
data.getVariableCount() # Return the number of computational variables in the sample
data.getVariables() # Returns a vector containing all the computational variables
data.getVariable(i) # Returns the value of the ith computational variable
data.getHashId() # Returns a unique identifier for the sample
```

In the code below shows how a computational model can access sample data:

```python
# This computational model sums the squares of all sample arguments.
def myModel(data):
 sum = 0.0;
 for i in range(data.getVariableCount()):
  sum += x.getVariable(i)*x.getVariable(i)
```

### Saving Results

Korali's engine requires the computational model to produce at least one result to be sent back to the Engine (depending on the evaluation type, it may require more than one result). To save the results of the evaluation, Korali exposes the following interface:

```python
data.addResult() # Adds a new result to the vector result to be sent back to Korali's engine
data.getResults() # Returns a the full results vector by reference
```

In the code below shows how a computational model saves a single result:

```python
# This computational model sums the squares of all sample arguments.
def myModel(data):
 sum = 0.0;
 for i in range(data.getVariableCount()):
  sum += x.getVariable(i)*x.getVariable(i)
 data.addResult(sum)
```

In the code below shows how a computational model saves a multiple results:

```python
# This computational model returns the squares for all sample arguments.
def myModel(data):
 for i in range(data.getVariableCount()):
  data.addResult(x.getVariable(i)*x.getVariable(i))
```