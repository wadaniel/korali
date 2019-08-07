# Constrained Covariance Matrix Adaptation Evolution Strategy 

## Description

### Base CCMAES

This solver also implements the *Constrained Covariance Matrix Adaptation Evolution Strategy*, as published in [Arampatzis2019](https://dl.acm.org/citation.cfm?doid=3324989.3325725).

In an evolution strategy, new candidate solutions are sampled according to a multivariate normal distribution in $\mathbb {R} ^{n}$. Recombination amounts to selecting a new mean value for the distribution. Mutation amounts to adding a random vector, a perturbation with zero mean. Pairwise dependencies between the variables in the distribution are represented by a covariance matrix. The covariance matrix adaptation (CMA) is a method to update the covariance matrix of this distribution.
CCMAES is an extension of [CCMAES](/usage/solvers/optimizers/cmaes/) for constrained optimization problems. It uses the principle of *viability boundaries* to find an initial mean vector for the proposal distribution that does not violate constraints, and secondly it uses a  *constraint handling technique* to efficiently adapt the proposal distribution to the constraints.


** Usage **

```python
  k["Solver"] = "CCMAES"
```

**Base Requirements:**

+ The *Sample Count* $\lambda$ needs to be defined.
+ The *Initial Mean* needs to be defined for every variable.
+ The *Initial Standard Deviation* needs to be defined for every variable.

** Constraint Requirements:**

+ At least one *Constraint function* defined.
+ The *Viability Sample Count* $via-\lambda$ needs to be defined.
+ The *Initial Mean* needs to be defined for every variable.
+ The *Initial Standard Deviation* needs to be defined for every variable.


## Solver Settings

??? abstract "Sample Count"

	Specifies the number of samples to evaluate per generation (preferably 
	4+3*log(N) number of variables).

	+ Default Value: 32
	+ Datatype: size_t
	+ Syntax: 

	```python
		korali["Solver"]["Sample Count"] = *value*
	```

??? abstract "Mu Value"

	Number of best samples used to update the covariance matrix and the mean.

	+ Default Value: _sampleCount*0.5
	+ Datatype: size_t
	+ Syntax: 

	```python
		korali["Solver"]["Mu Value"] = *value*
	```

??? abstract "Mu Type"

	Weights given to the Mu best values to update the covariance matrix and the mean.

	+ Default Value: "Logarithmic"
	+ Datatype: std::string
	+ Syntax: 

	```python
		korali["Solver"]["Mu Type"] = *value*
	```

??? abstract "Initial Sigma Cumulation Factor"

	Controls the learning rate of the conjugate evolution path.

	+ Default Value: -1.0
	+ Datatype: double
	+ Syntax: 

	```python
		korali["Solver"]["Initial Sigma Cumulation Factor"] = *value*
	```

??? abstract "Initial Damp Factor"

	Controls the updates of the covariance matrix scaling factor.

	+ Default Value: -1.0
	+ Datatype: double
	+ Syntax: 

	```python
		korali["Solver"]["Initial Damp Factor"] = *value*
	```

??? abstract "Is Sigma Bounded"

	Sets an upper bound for the covariance matrix scaling factor. The upper bound 
	is given by the average of the initial standard deviation of the variables

	+ Default Value: false
	+ Datatype: bool
	+ Syntax: 

	```python
		korali["Solver"]["Is Sigma Bounded"] = *value*
	```

??? abstract "Initial Cumulative Covariance"

	Controls the learning rate of the evolution path for the covariance update
	(must be in (0,1]).

	+ Default Value: -1.0
	+ Datatype: double
	+ Syntax: 

	```python
		korali["Solver"]["Initial Cumulative Covariance"] = *value*
	```

??? abstract "Is Diagonal"

	Covariance matrix updates will be optimized for diagonal matrices.

	+ Default Value: false
	+ Datatype: bool
	+ Syntax: 

	```python
		korali["Solver"]["Is Diagonal"] = *value*
	```

??? abstract "Viability Sample Count"

	Specifies the number of samples per generation during the viability 
	regime, i.e. during the search for a mean vector not violating the constraints.

	+ Default Value: 2
	+ Datatype: size_t
	+ Syntax: 

	```python
		korali["Solver"]["Viability Sample Count"] = *value*
	```

??? abstract "Viability Mu"

	Number of best samples used to update the covariance matrix and the mean 
	during the viability regime.

	+ Default Value: 1
	+ Datatype: size_t
	+ Syntax: 

	```python
		korali["Solver"]["Viability Mu"] = *value*
	```

??? abstract "Max Covariance Matrix Corrections"

	Max number of covairance matrix adaptions per generation during the constraint 
	handling loop.

	+ Default Value: 1e6
	+ Datatype: size_t
	+ Syntax: 

	```python
		korali["Solver"]["Max Covariance Matrix Corrections"] = *value*
	```

??? abstract "Target Success Rate"

	Controls the updates of the covariance matrix scaling factor during the
	viability regime.

	+ Default Value: 0.1818
	+ Datatype: double
	+ Syntax: 

	```python
		korali["Solver"]["Target Success Rate"] = *value*
	```

??? abstract "Covariance Matrix Adaption Strength"

	Controls the covariane matrix adaption strength if samples violate constraints.

	+ Default Value: 0.1
	+ Datatype: double
	+ Syntax: 

	```python
		korali["Solver"]["Covariance Matrix Adaption Strength"] = *value*
	```

??? abstract "Normal Vector Learning Rate"

	Learning rate of constraint normal vectors (must be in (0, 1]).

	+ Default Value: -1.0
	+ Datatype: double
	+ Syntax: 

	```python
		korali["Solver"]["Normal Vector Learning Rate"] = *value*
	```

??? abstract "Global Success Learning Rate"

	Learning rate of success probability of objective function improvements. 
	Required for covariance matrix scaling factor update during viability regime.

	+ Default Value: 0.2
	+ Datatype: double
	+ Syntax: 

	```python
		korali["Solver"]["Global Success Learning Rate"] = *value*
	```

## Variable Settings

??? abstract "Lower Bound"

	Specifies the lower bound for the variable's value. Korali will not generate samples
	for which this variable falls below the specified minimum. By default, Korali sets this
	value to -Infinity.

	+ Default Value: -std::numeric_limits<float>::max()
	+ Datatype: double
	+ Syntax: 

	```python
		korali["Variables"][i]["Lower Bound"] = *value*
	```

??? abstract "Upper Bound"

	Specifies the upper bound for the variable's value. Korali will not generate samples
	for which this variable falls below the specified maximum. By default, Korali sets this
	value to +Infinity.

	+ Default Value: +std::numeric_limits<float>::max()
	+ Datatype: double
	+ Syntax: 

	```python
		korali["Variables"][i]["Upper Bound"] = *value*
	```

??? abstract "Initial Mean"

	Defines the initial mean for the proposal distribution. This value must be defined
	between the variable's Mininum and Maximum settings By default, Korali sets this
	value in the center of the domain.

	+ Default Value: (_variableSettings[i].upperBound + _variableSettings[i].lowerBound)*0.5
	+ Datatype: double
	+ Syntax: 

	```python
		korali["Variables"][i]["Initial Mean"] = *value*
	```

??? abstract "Initial Standard Deviation"

	Defines the initial standard deviation of the proposal distribution for a variable.
	By default, Korali sets this value to 30% of the domain width.

	+ Default Value: (_variableSettings[i].upperBound - _variableSettings[i].lowerBound)*0.25
	+ Datatype: double
	+ Syntax: 

	```python
		korali["Variables"][i]["Initial Standard Deviation"] = *value*
	```

??? abstract "Minimum Standard Deviation Changes"

	Defines a lower bound for the standard deviation updates of the proposal distribution for a variable.
	Korali increases the scaling factor $\sigma$ if this value is undershot. By default,
	Korali sets this value to 0.0 (inactive).

	+ Default Value: 0.0
	+ Datatype: double
	+ Syntax: 

	```python
		korali["Variables"][i]["Minimum Standard Deviation Changes"] = *value*
	```

??? abstract "Discrete"

	Specifies if the variable can only take discrete values within the range of the 
	lower and upper bound.

	+ Default Value: false
	+ Datatype: bool
	+ Syntax: 

	```python
		korali["Variables"][i]["Discrete"] = *value*
	```

??? abstract "Granularity"

	Specifies the granulatiry of a discrete variable, a granularity of 1.0
	means that the variable can only take values in (.., -1.0, 0.0, +1.0, +2.0, ..)
	where the levels are set symmetric around the initial mean (here 0.0).

	+ Default Value: 0.0
	+ Datatype: double
	+ Syntax: 

	```python
		korali["Variables"][i]["Granularity"] = *value*
	```

## Termination Criteria

??? abstract "Max Infeasible Resampling"

	Number of resamplings per candidate per generation if sample is outside of 
	Lower and Upper Bound. 

	+ Default Value: 1e9
	+ Enabled by Default?: true
	+ Datatype: size_t
	+ Syntax: 

	```python
		korali["Solver"]["Termination Criteria"]["Max Infeasible Resampling"]["Value"] = *value*
	```

	```python
		korali["Solver"]["Termination Criteria"]["Max Infeasible Resampling"]["Enabled"] = *true / false* 
	```

??? abstract "Min Fitness"

	Specifies the target fitness to stop minimization.

	+ Default Value: -std::numeric_limits<float>::max()
	+ Enabled by Default?: false
	+ Datatype: double
	+ Syntax: 

	```python
		korali["Solver"]["Termination Criteria"]["Min Fitness"]["Value"] = *value*
	```

	```python
		korali["Solver"]["Termination Criteria"]["Min Fitness"]["Enabled"] = *true / false* 
	```

??? abstract "Max Fitness"

	Specifies the target fitness to stop maximization.

	+ Default Value: std::numeric_limits<float>::max()
	+ Enabled by Default?: false
	+ Datatype: double
	+ Syntax: 

	```python
		korali["Solver"]["Termination Criteria"]["Max Fitness"]["Value"] = *value*
	```

	```python
		korali["Solver"]["Termination Criteria"]["Max Fitness"]["Enabled"] = *true / false* 
	```

??? abstract "Min Fitness Diff Threshold"

	Specifies the minimum fitness differential between two consecutive generations 
	before stopping execution.

	+ Default Value: 1e-9
	+ Enabled by Default?: true
	+ Datatype: double
	+ Syntax: 

	```python
		korali["Solver"]["Termination Criteria"]["Min Fitness Diff Threshold"]["Value"] = *value*
	```

	```python
		korali["Solver"]["Termination Criteria"]["Min Fitness Diff Threshold"]["Enabled"] = *true / false* 
	```

??? abstract "Min Standard Deviation"

	Specifies the minimal standard deviation per dimension of the proposal.

	+ Default Value: 1e-12
	+ Enabled by Default?: false
	+ Datatype: double
	+ Syntax: 

	```python
		korali["Solver"]["Termination Criteria"]["Min Standard Deviation"]["Value"] = *value*
	```

	```python
		korali["Solver"]["Termination Criteria"]["Min Standard Deviation"]["Enabled"] = *true / false* 
	```

??? abstract "Max Standard Deviation"

	Specifies the maximal standard deviation per dimension of the proposal.

	+ Default Value: 1e18
	+ Enabled by Default?: false
	+ Datatype: double
	+ Syntax: 

	```python
		korali["Solver"]["Termination Criteria"]["Max Standard Deviation"]["Value"] = *value*
	```

	```python
		korali["Solver"]["Termination Criteria"]["Max Standard Deviation"]["Enabled"] = *true / false* 
	```

??? abstract "Max Condition Covariance Matrix"

	Specifies the maximum condition of the covariance matrix

	+ Default Value: 1e18
	+ Enabled by Default?: false
	+ Datatype: double
	+ Syntax: 

	```python
		korali["Solver"]["Termination Criteria"]["Max Condition Covariance Matrix"]["Value"] = *value*
	```

	```python
		korali["Solver"]["Termination Criteria"]["Max Condition Covariance Matrix"]["Enabled"] = *true / false* 
	```

??? abstract "Min Standard Deviation Step Factor"

	Specifies a scaling factor under which the standard deviation does not change
	in the direction of the eigenvectors.

	+ Default Value: 1e-18
	+ Enabled by Default?: false
	+ Datatype: double
	+ Syntax: 

	```python
		korali["Solver"]["Termination Criteria"]["Min Standard Deviation Step Factor"]["Value"] = *value*
	```

	```python
		korali["Solver"]["Termination Criteria"]["Min Standard Deviation Step Factor"]["Enabled"] = *true / false* 
	```

## Plotting

Plotting for CCMAES should include clear markings of rejected samples, or
showing the sample space that does not fit the constraints?


