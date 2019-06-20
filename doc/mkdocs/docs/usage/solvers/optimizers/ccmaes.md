# Solvers / CCMA-ES
   				   
##Description

This is the implementation of the *Constrained Covariance Matrix Adaptation Evolution Strategy*, as published in the [paper](https://dl.acm.org/citation.cfm?doid=3324989.3325725).

CCMA-ES is an extension of CMA-ES for constrained optimization problems. It uses
the principle of *viability boundaries* to find an initial mean vector for the proposal
distribution that does not violate constraints and secondly it uses an additional *constraint
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
  k["CCMA-ES"]["Termination Criteria"]["Max Generations"] ...
  k["CCMA-ES"]["Termination Criteria"]["Max Model Evaluations"] ...
  k["CCMA-ES"]["Termination Criteria"]["Min Fitness"] ...
  k["CCMA-ES"]["Termination Criteria"]["Fitness Diff Threshold"] ...
  k["CCMA-ES"]["Termination Criteria"]["Min DeltaX"] ...
  k["CCMA-ES"]["Termination Criteria"]["Max Standard Deviation"] ...
  k["CCMA-ES"]["Termination Criteria"]["Max Condition Covariance"] ...

  # Variable Settings
  k["Variables"][i]["CCMA-ES"]["Lower Bound"] = ...
  k["Variables"][i]["CCMA-ES"]["Upper Bound"] = ...
  k["Variables"][i]["CCMA-ES"]["Initial Mean"] = ...
  k["Variables"][i]["CCMA-ES"]["Initial Standard Deviation"] = ...
  k["Variables"][i]["CCMA-ES"]["Minimum Standard Deviation Changes"] = ...
  k["Variables"][i]["CCMA-ES"]["Log Space"] = ...
```

##CCMA-ES Specific Solver Settings

- **Viability Sample Count**. Specifies the number of samples ($via-\lambda$) used per generation in the viability regime, i.e. during the search for a mean vector $\bf{m}$ not violating the constraints. Recomended is a small population size. Example:

	```python
    # Setting viability lambda
	k["CCMA-ES"]["Viability Sample Count"] = 2

	```

- **Mu Value**. Number of best samples to update the covariance matrix and the mean during the viability regime. By default, Korali will set this value to $\lambda / 2$. Example:

	```python
    # Setting viability lambda
	k["CCMA-ES"]["Mu"]["Viability"] = 1

	```

- **Target Success Rate**. Controls the updates of the scaling factor $\sigma$ during viability regime. By default, Korali will set this value to $2/11$. Example:

	```python
    # Increasing Target Success Rate
	k["CCMA-ES"]["Target Success Rate"] = 0.6

	```

- **Adaption Size**. Controls the covariane matrix adaption strength if samples violate constraints. By default, Korali will set this value to $0.1$. Example:

	```python
    # Increasing Adaption Size
	k["CCMA-ES"]["Adaption Size"] = 0.3

	```

- **Max Corrections**. Max number of covairance matrix adaptions per generation during the constraint handling loop. By default, Korali will set this value to $10^6$. Example:

	```python
    # Setting viability lambda
	k["CCMA-ES"]["Max Corrections"] = 10e9

	```

- **Normal Vector Learning Rate**. Learning rate of constraint normal vectors (must be in $(0,1])$. By default, Korali will set this value to $\frac{1}{\lambda+2}$. Example:

	```python
    # Setting Normal Vector Learning Rate
	k["CCMA-ES"]["Normal Vector Learning Rate"] = 0.7

	```

- **Global Success Learning Rate**. Learning rate of success factor of objective function improvements. Required for scaling factor $\sigma$ update. By default, Korali will set this value to $\frac{1}{12}$. Example:

	```python
    # Setting viability lambda
	k["CCMA-ES"]["Global Success Learning Rate"] = 1

	```




##Solver Settings

- **Result Output Frequency**. Specifies the output frequency of intermediate result files. By default, Korali will set this value to *1* (i.e. each generation). Example:

	```python
    # Reduce the number of result files
	k["CCMA-ES"]["Result Output Frequency"] = 10

	```

- **Objective**. Specifies whether the problem evaluation is to be *minimized* or *maximized*. By default, Korali will set this value to *Maximize*. Example:

	```python
	#Maximizing problem evaluation (default)
	k["CCMA-ES"]["Objective"] = "Maximize"

	#Minimizing problem evaluation
	k["CCMA-ES"]["Objective"] = "Minimize"
	```
- **Sample Count**. Specifies the number of samples $\lambda$ to evaluate per each generation. Example:

	```python
	k["CCMA-ES"]["Sample Count"] = 32
	```
	
- **Sigma Cumulation Factor**. Controls the learning rate of the conjugate evolution path for $\sigma$ update. By default, Korali will set this value to $\frac{\mu_{eff}+2}{\mu_{eff}+N+3}$. Example:

	```python
	k["CCMA-ES"]["Sigma Cumulation Factor"] = 1.0
	```

- **Damp Factor**. Controls the update of the scaling factor $\sigma$. By default, Korali will precalibrate this value from $\mu_{eff}$ and number of variables (must be larger 0.0). Example:

	```python
	k["CCMA-ES"]["Damp Factor"] = 1.0
	```	
	
- **Max Resamplings**. Number of resamplings per candidate per generation if sample is outside of Lower and Uppder Bound. By default, Korali will set this value to $10^6$. Example:

	```python
	k["CCMA-ES"]["Max Resamplings"] = 10e6
	```	

- **Sigma Bounded**. Sets an upper bound for scaling factor $\sigma$. Upper bound is given by the average of the initial standard deviation of the variables. By default, Korali will set this value to *False*. Example:

	```python
	k["CCMA-ES"]["Sigma Bounded"] = True
	```	
	
- **Mu Value**. Number of best samples to update the covariance matrix and the mean. By default, Korali will set this value to $\lambda / 2$. Example:

	```python
	k["CCMA-ES"]["Mu"]["Value"] = 8
	```	

- **Mu Type**. Weights given to the Mu best values to update the covariance matrix and the mean. By default, Korali will set this value to *Logarithmic*. Example:

	```python
	k["CCMA-ES"]["Mu"]["Type"] = "Equal"
	k["CCMA-ES"]["Mu"]["Type"] = "Linear"
	k["CCMA-ES"]["Mu"]["Type"] = "Logarithmic"
	```

- **Cumulative Covariance**. Controls the learning rate of the evolution path for the covariance update. By default, Korali will precalibrate this value from $\mu_{eff}$ and number of variables (must be in $(0,1]$). Example:

	```python
	k["CCMA-ES"]["Covariance Matrix"]["Cumulative Covariance"] = 0.9
	```

- **Is Diag**. Tells the solver to treat the covariance matrix as diagonal matrix in order to speed up covariance matrix updates. By default, Korali will set this value to *False*. Example:

	```python
	k["CCMA-ES"]["Covariance Matrix"]["Is Diag"] = True
	```


		 
## Termination Criteria

- **Max Generations** Specifies the maximum number of generations to run. By default, Korali will set this criterion as active and its value to *1000*. Example:

	```python
	k["CCMA-ES"]["Termination Criteria"]["Max Generations"]["Active"] = True
	k["CCMA-ES"]["Termination Criteria"]["Max Generations"]["Value"]  = 2000
	```

	
- **Max Model Evaluations** Specifies the maximum number of objective function evaluations. By default, Korali will set this criterion as inactive. Example:

	```python
	k["CCMA-ES"]["Termination Criteria"]["Max Model Evaluations"]["Active"] = True
	k["CCMA-ES"]["Termination Criteria"]["Max Model Evaluations"]["Value"]  = 50000
	```	
	
- **Fitness** Specifies the goal fitness to achieve before stopping execution. By default, Korali will set this criterion as inactive. Example:

	```python
	k["CCMA-ES"]["Termination Criteria"]["Min Fitness"]["Active"] = True
	k["CCMA-ES"]["Termination Criteria"]["Min Fitness"]["Value"]  = 0.0
	```	
	
- **Fitness Diff Threshold** Specifies the minimum fitness differential between two consecutive generations before stopping execution. By default, Korali will set this criterion as active and value $1^-9$. Example:

	```python
	k["CCMA-ES"]["Termination Criteria"]["Fitness Diff Threshold"]["Active"] = True
	k["CCMA-ES"]["Termination Criteria"]["Fitness Diff Threshold"]["Value"]  = 1e-12
	```	

- **Min Standard Deviation** Specifies a minimum standard deviation for the proposal distribution of the variables. Korali terminates if this value is undershot from all variables. By default, Korali will set this criterion as active and value $1^-12$. Example:

	```python
	k["CCMA-ES"]["Termination Criteria"]["Min DeltaX"]["Active"] = True
	k["CCMA-ES"]["Termination Criteria"]["Min DeltaX"]["Value"]  = 1e-9
	```	
	
- **Max Standard Deviation** Specifies the maximum allowed increase in the standard deviation for the proposal distribution of a variable. By default, Korali will set this criterion as active and value $1^18$. Example:

	```python
	k["CCMA-ES"]["Termination Criteria"]["Max Standard Deviation"]["Active"] = True
	k["CCMA-ES"]["Termination Criteria"]["Max Standard Deviation"]["Value"]  = 1e12
	```	

- **Max Condition Covariance** Specifies the maximum condition of the covariance matrix. By default, Korali will set this criterion as active and value $1^18$. Example:

	```python
	k["CCMA-ES"]["Termination Criteria"]["Max Standard Deviation"]["Active"] = True
	k["CCMA-ES"]["Termination Criteria"]["Max Standard Deviation"]["Value"]  = 1e12
	```	
	
## Variable Settings

- **Minimum** Specifies the lower bound for the variable's value. Korali will not generate samples in which this variable falls below the specified minimum. By default, Korali will set this value to *-Infinity*. Example:

	```python
	# Modifying the lower bound of two variables
	k["Variables"][i]["CCMA-ES"][0]["Minimum"] = 0.0;
	k["Variables"][i]["CCMA-ES"][1]["Minimum"] = -32.0;
	```

- **Maximum** Specifies the upper bound for the variable's value. Korali will not generate samples in which this variable falls below the specified minimum. By default, Korali will set this value to *+Infinity*. Example:

	```python
	# Modifying the upper bound of two variables
	k["Variables"][i]["CCMA-ES"][0]["Maximum"] = 32.0;
	k["Variables"][i]["CCMA-ES"][1]["Maximum"] = 0.0;
	```

- **Initial Mean**. Defines the initial mean for the proposal distribution. This value must be defined between the variable's *Mininum* and *Maximum* settings By default, Korali will set this value in the center of the domain. Example:

	```python
	# Modifying the initial guess mean of variable i
	k["Variables"][i]["CCMA-ES"]["Initial Mean"] = 16.0;
	```

- **Initial Standard Deviation**. Defines the initial standard deviation of the proposal distribution for a variable. By default, Korali sets this value to 30% of the domain width. Example:

	```python
	# Modifying the initial standard deviation of my variable
	k["Variables"][i]["CCMA-ES"]["Initial Standard Deviation"] = 2.0;
	```

- **Minimum Standard Deviation Changes**. Defines a lower bound for the standard deviation of the proposal distribution for a variable. Korali increases the scaling factor $\sigma$ if this value is undershot. By default, Korali sets this value to 0.0 (inactive). Example:

	```python
	# Modifying the initial standard deviation of my variable
	k["Variables"][i]["CCMA-ES"]["Minimum Standard Deviation Changes"] = 1e-6
	```

- **Log Space**. Specifies if the log space of a variable should be searched. By default, Korali sets this value to $False$.
		
    ```python
	# Search in log space
	k["Variables"][i]["CCMA-ES"]["Log Space"] = True
	```


## Plotting

For plotting we use the same tool as in [CMA-ES](TODO), for more information
please refer given page. 

