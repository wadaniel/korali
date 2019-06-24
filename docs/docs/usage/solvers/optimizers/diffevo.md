# Solvers / DE
   				   
##Description

This is the implementation of the *Differential Evolution Algorithm*, as published in [Storn1997](https://link.springer.com/article/10.1023/A:1008202821328).

DE optimizes a problem by updating a population of candidate solutions through $mutation$ and
$recombination$. The update rules are simple and the objective function must not
be differentiable.
Our implementation includes various adaption and updating strategies
[Brest2006](https://ieeexplore.ieee.org/document/4016057).

**Requirements:**

+ The *Sample Count* needs to be defined.
+ The *Lower Bound* needs to be defined for every variable.
+ The *Upper Bound* needs to be defined for every variable.

##Settings

For a better understanding of the variables please refer to the paper.

```python
  # Definition
  k["Solver"] = "DE";
  
  k["DE"]["Result Output Frequency"] = ...
  
  # Solver Settings
  k["DE"]["Objective"] = ... 
  k["DE"]["Sample Count"] = ...
  k["DE"]["Crossover Rate"] = ...
  k["DE"]["Mutation Rate"] = ...
  k["DE"]["Mutation Rule"] = ...
  k["DE"]["Parent"] = ...
  k["DE"]["Max Resamplings"] = ...
  k["DE"]["Accept Rule"] = ...
  k["DE"]["Fix Infeasible"] = ...
  
  # Termination Criteria
  k["DE"]["Termination Criteria"]["Max Generations"]["Active"] = ...
  k["DE"]["Termination Criteria"]["Max Generations"]["Value"] = ...
  k["DE"]["Termination Criteria"]["Max Model Evaluations"]["Active"] = ...
  k["DE"]["Termination Criteria"]["Max Model Evaluations"]["Value"] = ...
  k["DE"]["Termination Criteria"]["Fitness"]["Active"] = ...
  k["DE"]["Termination Criteria"]["Fitness"]["Value"] = ...
  k["DE"]["Termination Criteria"]["Fitness Diff Threshold"]["Active"] = ...
  k["DE"]["Termination Criteria"]["Fitness Diff Threshold"]["Value"] = ...
  k["DE"]["Termination Criteria"]["Min DeltaX"]["Active"] = ...
  k["DE"]["Termination Criteria"]["Min DeltaX"]["Value"] = ...

  # Variable Settings
  k["Variables"][i]["DE"]["Lower Bound"] = ...
  k["Variables"][i]["DE"]["Upper Bound"] = ...
  k["Variables"][i]["DE"]["Initial Mean"] = ...
  k["Variables"][i]["DE"]["Initial Standard Deviation"] = ...
  k["Variables"][i]["DE"]["Log Space"] = ...
```


##Solver Settings

- **Result Output Frequency**. Specifies the output frequency of intermediate result files. By default, Korali sets this value to *1* (i.e. each generation). Example:

	```python
    # Reduce the number of result files
	k["DE"]["Result Output Frequency"] = 10

	```

- **Objective**. Specifies whether the problem evaluation is to be *minimized* or *maximized*. By default, Korali sets this value to *Maximize*. Example:

	```python
	#Maximizing problem evaluation (default)
	k["DE"]["Objective"] = "Maximize"

	#Minimizing problem evaluation
	k["DE"]["Objective"] = "Minimize"
	```

- **Sample Count**. Specifies the number of samples to evaluate per generation (preferably 5-10x number of variables). Example:

	```python
    # Specifying Sample Count
	k["DE"]["Sample Count"] = 20
	```

- **Crossover Rate**. Controls the rate at which dimensions from samples are mixed (must be in $[0,1]$). By default, Korali sets thi value *0.9*. Example:

	```python
    # Increasing Crossover Rate
	k["DE"]["Crossover Rate"] = 1.0
	```	
	
- **Mutation Rate**. Controls the scaling of the vector differentials (must be in $[0,2]$, preferably $<1$). By default, Korali sets this value to *0.1*. Example:

	```python
    # Increasing Mutation Rate
	k["DE"]["Mutation Rate"] = 0.4
	```	

- **Mutation Rule**. Controls the Mutation Rate: "Default" (rate is fixed) or "Self Adaptive" (udpating rule in [Brest2006]). By default, Korali sets this value to *Default*. Example:

	```python
    # Increasing Mutation Rate
	k["DE"]["Mutation Rate"] = 0.4
	```	

- **Parent**. Controls the selection of the parent vecor: "Random" or "Best" (best sample from previous generation). By default, Korali sets this value to *Random*. Example:

	```python
    # Select Best vector as Parent for all mutations
	k["DE"]["Parent"] = "Best"
	```


- **Accept Rule**. Sets the accept rule after mutation: "Best", "Greedy", "Iterative" or "Improved". By default, Korali setis this value to *Greedy*. Example:

	```pythona
    # Specify the Accept Rule
	k["DE"]["Accept Rule"] = "Iterative"
	```	
	
- **Fix Infeasible**. If set true, Korali samples a random sample between Parent and the voiolated boundary. If set false, infeasible samples are mutated again until feasible. By default, Korali sets this flag to *True*. Example:

	```python
    # Disable Fix Infeasible
	k["DE"]["Mu"]["Fix Infeasible"] = False
	```	

- **Max Resampling**. Max number of mutations per sample per generation if infeasible (only relevant if Fix Infeasible is set False). By default, Korali sets this value to $10^6$. Example:

	```python
    # Increase Max Resampling
	k["DE"]["Max Resampling"] = 1e9
	```
		 
## Termination Criteria

- **Max Generations** Specifies the maximum number of generations to run. By default, Korali sets this criterion as active and its value to *1000*. Example:

	```python
    # Increase number of Generations to run
	k["DE"]["Termination Criteria"]["Max Generations"]["Active"] = True
	k["DE"]["Termination Criteria"]["Max Generations"]["Value"]  = 5000
	```

	
- **Max Model Evaluations** Specifies the maximum number of objective function evaluations. By default, Korali sets this criterion as inactive. Example:

	```python
    # Specifying Max Model Evaluations
	k["DE"]["Termination Criteria"]["Max Model Evaluations"]["Active"] = True
	k["DE"]["Termination Criteria"]["Max Model Evaluations"]["Value"]  = 50000
	```	
	
- **Fitness** Specifies the goal fitness to stop execution. By default, Korali sets this criterion as inactive. Example:

	```python
    # Specifying target Fitness
	k["DE"]["Termination Criteria"]["Fitness"]["Active"] = True
	k["DE"]["Termination Criteria"]["Fitness"]["Value"]  = 0.0
	```	
	
- **Fitness Diff Threshold** Specifies the minimum fitness differential of the best samples between two consecutive generations before stopping execution. By default, Korali sets this criterion inactive. Example:

	```python
    # Specifying Fitness Diff Threshold
	k["DE"]["Termination Criteria"]["Fitness Diff Threshold"]["Active"] = True
	k["DE"]["Termination Criteria"]["Fitness Diff Threshold"]["Value"]  = 1e-12
	```	

- **Min DeltaX** Specifies the minimum change of the mean of the samples before stopping execution. By default, Korali sets this criterion as inactive. Example:

	```python
    # Specifying Min DeltaX
	k["DE"]["Termination Criteria"]["Min DeltaX"]["Active"] = True
	k["DE"]["Termination Criteria"]["Min DeltaX"]["Value"]  = 1e-9
	```	
	
## Variable Settings

- **Name** Specifies a Name for the variables, only used for output. By default, Korali sets this value to $Xi$. Example:

	```python
	# Specify Variable  Names
	k["Variables"][0]["DE"]["Name"] = "Theta";
	k["Variables"][1]["DE"]["Name"] = "Sigma";
	```

- **Lower Bound** Specifies the lower bound for the variable's value. Korali will not generate samples for which this variable falls below the specified minimum. Example:

	```python
	# Modifying the lower bound of two variables
	k["Variables"][0]["DE"]["Lower Bound"] = -32.0
	k["Variables"][1]["DE"]["Lower Bound"] = -32.0;
	```

- **Upper Bound** Specifies the upper bound for the variable's value. Korali will not generate samples for which this variable falls above the specified maximum. Example:

	```python
	# Modifying the upper bound of two variables
	k["Variables"][0]["DE"]["Upper Bound"] = 32.0;
	k["Variables"][1]["DE"]["Upper Bound"] = 32.0;
	```

- **Initial Mean**. Defines the initial mean for the multivariate proposal distribution to initialize the samples. By default, Korali does *not* set this value and samples are drawn uniformly from the valid domain. Example:

	```python
	# Modifying the Initial Mean ean of variable 0 and 1
	k["Variables"][0]["DE"]["Initial Mean"] = 16.0;
	k["Variables"][1]["DE"]["Initial Mean"] = 16.0;
	```

- **Initial Standard Deviation**. Defines the initial standard deviation of the multivariate proposal distribution to initialize the samples. By default, Korali does *not* set this value and samples are drawn uniformly fron the valid domain. Example:

	```python
	# Modifying the Initial Standard Deviation of variable 0 and 1
	k["Variables"][0]["DE"]["Initial Standard Deviation"] = 2.0;
	k["Variables"][1]["DE"]["Initial Standard Deviation"] = 2.0;
	```

- **Log Space**. Specifies if the log space of a variable should be searched. By default, Korali sets this flag to $False$.
		
    ```python
	# Search in Log Space
	k["Variables"][0]["DE"]["Log Space"] = False
	k["Variables"][1]["DE"]["Log Space"] = True
	```

