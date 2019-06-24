# Solvers / CCMA-ES
   				   
##Description

This is the implementation of the *Constrained Covariance Matrix Adaptation Evolution Strategy*, as published in [Arampatzis2019](https://dl.acm.org/citation.cfm?doid=3324989.3325725).

CCMA-ES is an extension of [CMA-ES](/usage/solvers/optimizers/cmaes/) for constrained optimization problems. It uses
the principle of *viability boundaries* to find an initial mean vector for the proposal
distribution that does not violate constraints, and secondly it uses a  *constraint
handling technique* to efficiently adapt the proposal distribution to the
constraints.

**Requirements:**

+ *Constraint functions*
+ The *Sample Count* $\lambda$ needs to be defined.
+ The *Viability Sample Count* $via-\lambda$ needs to be defined.
+ The *Initial Mean* needs to be defined for every variable.
+ The *Initial Standard Deviation* needs to be defined for every variable.

##Settings

For a better understanding of the variables please refer to the paper.

```python
  # Definition
  k["Solver"] = "CCMA-ES";
  
  k["CCMA-ES"]["Result Output Frequency"] = ...
  
  # CCMA-ES Specific Solver Settings
  k["CCMA-ES"]["Viability Sample Count"] ...
  k["CCMA-ES"]["Mu"]["Viability"] ...
  k["CCMA-ES"]["Target Success Rate"] ...
  k["CCMA-ES"]["Adaption Size"] ...
  k["CCMA-ES"]["Max Corrections"] ...
  k["CCMA-ES"]["Normal Vector Learning Rate"] ...
  k["CCMA-ES"]["Global Success Learnin Rate"] ...

  # Solver Settings
  k["CCMA-ES"]["Objective"] = ... 
  k["CCMA-ES"]["Sample Count"] = ...
  k["CCMA-ES"]["Sigma Cumulation Factor"] = ...
  k["CCMA-ES"]["Damp Factor"] = ...
  k["CCMA-ES"]["Max Resamplings"] = ...
  k["CCMA-ES"]["Sigma Bounded"] = ...
  
  k["CCMA-ES"]["Mu"]["Value"] = ...
  k["CCMA-ES"]["Mu"]["Type"] = ...
  k["CCMA-ES"]["Mu"]["Covariance"] = ...

  k["CCMA-ES"]["Covariance Matrix"]["Cumulative Covariance"] = ...
  k["CCMA-ES"]["Covariance Matrix"]["Learning Rate"] = ...
  k["CCMA-ES"]["Covariance Matrix"]["Eigenvalue Evaluation Frequency"] = ...
  k["CCMA-ES"]["Covariance Matrix"]["Is Diagonal"] = ...      

  # Termination Criteria
  k["CMA-ES"]["Termination Criteria"]["Max Generations"]["Active"] = ...
  k["CMA-ES"]["Termination Criteria"]["Max Generations"]["Value"] = ...
  k["CMA-ES"]["Termination Criteria"]["Max Model Evaluations"]["Active"] = ...
  k["CMA-ES"]["Termination Criteria"]["Max Model Evaluations"]["Value"] = ...
  k["CMA-ES"]["Termination Criteria"]["Min Fitness"]["Active"] = ...
  k["CMA-ES"]["Termination Criteria"]["Min Fitness"]["Value"] = ...
  k["CMA-ES"]["Termination Criteria"]["Fitness Diff Threshold"]["Active"] =  ...
  k["CMA-ES"]["Termination Criteria"]["Fitness Diff Threshold"]["Value"] =  ...
  k["CMA-ES"]["Termination Criteria"]["Min DeltaX"]["Active"] = ...
  k["CMA-ES"]["Termination Criteria"]["Min DeltaX"]["Value"] = ...
  k["CMA-ES"]["Termination Criteria"]["Max Standard Deviation"]["Active"] = ...
  k["CMA-ES"]["Termination Criteria"]["Max Standard Deviation"]["Value"] = ...
  k["CMA-ES"]["Termination Criteria"]["Max Condition Covariance"]["Active"] = ...
  k["CMA-ES"]["Termination Criteria"]["Max Condition Covariance"]["Value"] = ...

  # Variable Settings
  k["Variables"][i]["CCMA-ES"]["Lower Bound"] = ...
  k["Variables"][i]["CCMA-ES"]["Upper Bound"] = ...
  k["Variables"][i]["CCMA-ES"]["Initial Mean"] = ...
  k["Variables"][i]["CCMA-ES"]["Initial Standard Deviation"] = ...
  k["Variables"][i]["CCMA-ES"]["Minimum Standard Deviation Changes"] = ...
  k["Variables"][i]["CCMA-ES"]["Log Space"] = ...
```

##CCMA-ES Specific Solver Settings

- **Viability Sample Count**. Specifies the number of samples ($via-\lambda$) used per generation during the viability regime, i.e. during the search for a mean vector $\bf{m}$ not violating the constraints. Recomended is a small population size. Example:

	```python
    # Setting Viability Sample Count
	k["CCMA-ES"]["Viability Sample Count"] = 2

	```

- **Mu Viability**. Number of best samples used to update the covariance matrix and the mean during the viability regime. By default, Korali sets this value to $via-\lambda / 2$. Example:

	```python
    # Setting Mu Viability
	k["CCMA-ES"]["Mu"]["Viability"] = 1

	```

- **Target Success Rate**. Controls the updates of the scaling factor $\sigma$ during the viability regime. By default, Korali sets this value to $2/11$. Example:

	```python
    # Increasing Target Success Rate
	k["CCMA-ES"]["Target Success Rate"] = 0.6

	```

- **Adaption Size**. Controls the covariane matrix adaption strength if samples violate constraints. By default, Korali sets this value to $0.1$. Example:

	```python
    # Increasing Adaption Size
	k["CCMA-ES"]["Adaption Size"] = 0.3

	```

- **Max Corrections**. Max number of covairance matrix adaptions per generation during the constraint handling loop. By default, Korali sets this value to $10^6$. Example:

	```python
    # Increasing Max Corrections
	k["CCMA-ES"]["Max Corrections"] = 10e9

	```

- **Normal Vector Learning Rate**. Learning rate of constraint normal vectors (must be in $(0,1])$. By default, Korali sets this value to $\frac{1}{\lambda+2}$. Example:

	```python
    # Increasing Normal Vector Learning Rate
	k["CCMA-ES"]["Normal Vector Learning Rate"] = 0.7

	```

- **Global Success Learning Rate**. Learning rate of success factor of objective function improvements. Required for scaling factor $\sigma$ update. By default, Korali sets this value to $\frac{1}{12}$. Example:

	```python
    # Increasing Global Success Learning Rate
	k["CCMA-ES"]["Global Success Learning Rate"] = 0.5

	```

##Solver Settings

- **Result Output Frequency**. Specifies the output frequency of intermediate result files. By default, Korali sets this value to *1* (i.e. each generation). Example:

	```python
    # Reduce the number of result files
	k["CCMA-ES"]["Result Output Frequency"] = 10

	```

- **Objective**. Specifies whether the problem evaluation is to be *minimized* or *maximized*. By default, Korali sets this value to *Maximize*. Example:

	```python
	#Maximizing problem evaluation (default)
	k["CCMA-ES"]["Objective"] = "Maximize"

	#Minimizing problem evaluation
	k["CCMA-ES"]["Objective"] = "Minimize"
	```

- **Sample Count**. Specifies the number of samples $\lambda$ to evaluate per generation. Example:

	```python
    # Specifying Sample Count
	k["CCMA-ES"]["Sample Count"] = 32
	```

- **Sigma Cumulation Factor**. Controls the learning rate of the conjugate evolution path for $\sigma$ update. By default, Korali sets this value to $\frac{\mu_{eff}+2}{\mu_{eff}+N+3}$. Example:

	```python
    # Increasing Sigma Cumulation Factor
	k["CCMA-ES"]["Sigma Cumulation Factor"] = 1.0
	```

- **Damp Factor**. Controls the update of the scaling factor $\sigma$. By default, Korali calibrates this value from $\mu_{eff}$ and number of variables (must be larger 0.0). Example:

	```python
    # Increasing Damp Factor
	k["CCMA-ES"]["Damp Factor"] = 1.0
	```	
	
- **Max Resamplings**. Number of resamplings per candidate per generation if sample is outside of Lower and Uppder Bound. By default, Korali sets this value to $10^6$. Example:

	```python
    # Increasing Max Resamplings
	k["CCMA-ES"]["Max Resamplings"] = 10e7
	```	

- **Sigma Bounded**. Sets an upper bound for scaling factor $\sigma$. The upper bound is given by the average of the initial standard deviation of the variables. By default, Korali setis this flag to *False*. Example:

	```pythona
    # Set an upper limit for Sigma
	k["CCMA-ES"]["Sigma Bounded"] = True
	```	
	
- **Mu Value**. Number of best samples used to update the covariance matrix and the mean. By default, Korali sets this value to $\lambda / 2$. Example:

	```python
    # Specifying Mu
	k["CCMA-ES"]["Mu"]["Value"] = 8
	```	

- **Mu Type**. Weights given to the Mu best values to update the covariance matrix and the mean. By default, Korali sets this value to *Logarithmic*. Example:

	```python
	k["CCMA-ES"]["Mu"]["Type"] = "Equal"
	k["CCMA-ES"]["Mu"]["Type"] = "Linear"
	k["CCMA-ES"]["Mu"]["Type"] = "Logarithmic"
	```

- **Cumulative Covariance**. Controls the learning rate of the evolution path for the covariance update. By default, Korali calibrates this value from $\mu_{eff}$ and number of variables (must be in $(0,1]$). Example:

	```python
    # Increasing Cumulative Covariance
	k["CCMA-ES"]["Covariance Matrix"]["Cumulative Covariance"] = 0.9
	```

- **Is Diag**. Tells the solver to treat the covariance matrix as diagonal in order to speed up covariance matrix updates. By default, Korali sets this flag to *False*. Example:

	```python
    # Diagonal Covariance Matrix
	k["CCMA-ES"]["Covariance Matrix"]["Is Diag"] = True
	```


		 
## Termination Criteria

- **Max Generations** Specifies the maximum number of generations to run. By default, Korali sets this criterion as active and its value to *1000*. Example:

	```python
    # Increase number of Generations to run
	k["CCMA-ES"]["Termination Criteria"]["Max Generations"]["Active"] = True
	k["CCMA-ES"]["Termination Criteria"]["Max Generations"]["Value"]  = 5000
	```

	
- **Max Model Evaluations** Specifies the maximum number of objective function evaluations. By default, Korali sets this criterion as inactive. Example:

	```python
    # Specifying Max Model Evaluations
	k["CCMA-ES"]["Termination Criteria"]["Max Model Evaluations"]["Active"] = True
	k["CCMA-ES"]["Termination Criteria"]["Max Model Evaluations"]["Value"]  = 50000
	```	
	
- **Fitness** Specifies the goal fitness to stop execution. By default, Korali sets this criterion as inactive. Example:

	```python
    # Specifying Min Fitness
	k["CCMA-ES"]["Termination Criteria"]["Min Fitness"]["Active"] = True
	k["CCMA-ES"]["Termination Criteria"]["Min Fitness"]["Value"]  = 0.0
	```	
	
- **Fitness Diff Threshold** Specifies the minimum fitness differential between two consecutive generations before stopping execution. By default, Korali sets this criterion as active and value $1^-9$. Example:

	```python
    # Decreasing Fitness Diff Threshold
	k["CCMA-ES"]["Termination Criteria"]["Fitness Diff Threshold"]["Active"] = True
	k["CCMA-ES"]["Termination Criteria"]["Fitness Diff Threshold"]["Value"]  = 1e-12
	```	

- **Min Standard Deviation** Specifies a minimum standard deviation for the proposal distribution of the variables. Korali terminates if this value is undershot by all variables. By default, Korali sets this criterion as active and value $1^-12$. Example:

	```python
    # Increasing Min Standard Deviation
	k["CCMA-ES"]["Termination Criteria"]["Min Standard Deviation"]["Active"] = True
	k["CCMA-ES"]["Termination Criteria"]["Min Standard Deviation"]["Value"]  = 1e-9
	```	
	
- **Max Standard Deviation** Specifies the maximum allowed increase in the standard deviation for the proposal distribution of a variable. By default, Korali sets this criterion as active and value $1^18$. Example:

	```python
    # Decreasing Max Standard Deviation
	k["CCMA-ES"]["Termination Criteria"]["Max Standard Deviation"]["Active"] = True
	k["CCMA-ES"]["Termination Criteria"]["Max Standard Deviation"]["Value"]  = 1e12
	```	

- **Max Condition Covariance** Specifies the maximum condition of the covariance matrix. By default, Korali sets this criterion as active and value $1^18$. Example:

	```python
    # Decreasing Max Standard Deviation
	k["CCMA-ES"]["Termination Criteria"]["Max Condition Covariance"]["Active"] = True
	k["CCMA-ES"]["Termination Criteria"]["Max Condition Covariance"]["Value"]  = 1e15
	```	
	
## Variable Settings

- **Name** Specifies a Name for the variables, only used for output. By default, Korali sets this value to $Xi$. Example:

	```python
	# Specify Variable  Names
	k["Variables"][0]["CCMA-ES"]["Name"] = "Theta";
	k["Variables"][1]["CCMA-ES"]["Name"] = "Sigma";
	```

- **Minimum** Specifies the lower bound for the variable's value. Korali will not generate samples for which this variable falls below the specified minimum. By default, Korali sets this value to *-Infinity*. Example:

	```python
	# Modifying the lower bound of two variables
	k["Variables"][0]["CCMA-ES"]["Minimum"] = -32.0
	k["Variables"][1]["CCMA-ES"]["Minimum"] = -32.0;
	```

- **Maximum** Specifies the upper bound for the variable's value. Korali will not generate samples for which this variable falls below the specified maximum. By default, Korali sets this value to *+Infinity*. Example:

	```python
	# Modifying the upper bound of two variables
	k["Variables"][0]["CCMA-ES"]["Maximum"] = 32.0;
	k["Variables"][1]["CCMA-ES"]["Maximum"] = 32.0;
	```

- **Initial Mean**. Defines the initial mean for the proposal distribution. This value must be defined between the variable's *Mininum* and *Maximum* settings By default, Korali sets this value in the center of the domain. Example:

	```python
	# Modifying the Initial Mean ean of variable 0 and 1
	k["Variables"][0]["CCMA-ES"]["Initial Mean"] = 16.0;
	k["Variables"][1]["CCMA-ES"]["Initial Mean"] = 16.0;
	```

- **Initial Standard Deviation**. Defines the initial standard deviation of the proposal distribution for a variable. By default, Korali sets this value to 30% of the domain width. Example:

	```python
	# Modifying the Initial Standard Deviation of variable 0 and 1
	k["Variables"][0]["CCMA-ES"]["Initial Standard Deviation"] = 2.0;
	k["Variables"][1]["CCMA-ES"]["Initial Standard Deviation"] = 2.0;
	```

- **Minimum Standard Deviation Changes**. Defines a lower bound for the standard deviation updates of the proposal distribution for a variable. Korali increases the scaling factor $\sigma$ if this value is undershot. By default, Korali sets this value to 0.0 (inactive). Example:

	```python
	# Specifying Minum Standard Deviation Changes
	k["Variables"][0]["CCMA-ES"]["Minimum Standard Deviation Changes"] = 1e-6
	k["Variables"][1]["CCMA-ES"]["Minimum Standard Deviation Changes"] = 1e-6
	```

- **Log Space**. Specifies if the log space of a variable should be searched. By default, Korali sets this flag to $False$.
		
    ```python
	# Search in Log Space
	k["Variables"][i]["CCMA-ES"]["Log Space"] = True
	```


## Plotting

For plotting we use the same tool as in [CMA-ES](/usage/solvers/optimizers/cmaes/#plotting), for more information
please refer to given page. 

