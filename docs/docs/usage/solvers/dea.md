# Differential Evolution Algorithm 

## Description

his is an implementation of the *Differential Evolution Algorithm* algorithm,
as published in [Storn1997](https://link.springer.com/article/10.1023/A:1008202821328.

DEA optimizes a problem by updating a population of candidate solutions through
mutation and recombination. The update rules are simple and the objective 
function must not be differentiable. Our implementation includes various adaption 
and updating strategies [Brest2006](https://ieeexplore.ieee.org/document/4016057).

** Usage **

```python
  k["Solver"] = "DEA"
```
	
**Requirements:**

+ The *Sample Count* needs to be defined..
+ The *Lower Bound* needs to be defined for each variable.
+ The *Upper Bound* needs to be defined for every variable.


## Solver Settings

??? abstract "Sample Count"

	Specifies the number of samples to evaluate per generation (preferably 5-10x 
	number of variables).

	+ Default Value: 
	+ Datatype: size_t
	+ Syntax: 

	```python
		korali["Solver"]["Sample Count"] = *value*
	```

??? abstract "Crossover Rate"

	Controls the rate at which dimensions from samples are mixed (must be in [0,1]).

	+ Default Value: 0.9
	+ Datatype: double
	+ Syntax: 

	```python
		korali["Solver"]["Crossover Rate"] = *value*
	```

??? abstract "Mutation Rate"

	Controls the scaling of the vector differentials (must be in [0,2], preferably < 1).

	+ Default Value: 0.5
	+ Datatype: double
	+ Syntax: 

	```python
		korali["Solver"]["Mutation Rate"] = *value*
	```

??? abstract "Mutation Rule"

	Controls the Mutation Rate: "Default" (rate is fixed) or "Self Adaptive" 
	(udpating rule in [Brest2006]).

	+ Default Value: "Default"
	+ Datatype: std::string
	+ Syntax: 

	```python
		korali["Solver"]["Mutation Rule"] = *value*
	```

??? abstract "Parent Selection Rule"

	Controls the selection of the parent vecor: "Random" or "Best" (best sample 
	from previous generation).

	+ Default Value: "Random"
	+ Datatype: std::string
	+ Syntax: 

	```python
		korali["Solver"]["Parent Selection Rule"] = *value*
	```

??? abstract "Accept Rule"

	Sets the sample accept rule after mutation and evaluation: "Best", "Greedy", 
	"Iterative" or "Improved".

	+ Default Value: "Greedy"
	+ Datatype: std::string
	+ Syntax: 

	```python
		korali["Solver"]["Accept Rule"] = *value*
	```

??? abstract "Fix Infeasible"

	If set true, Korali samples a random sample between Parent and the voiolated 
	boundary. If set false, infeasible samples are mutated again until feasible.

	+ Default Value: true
	+ Datatype: bool
	+ Syntax: 

	```python
		korali["Solver"]["Fix Infeasible"] = *value*
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

## Termination Criteria

??? abstract "Max Resamplings"

	Max number of mutations per sample per generation if infeasible (only relevant 
	if Fix Infeasible is set False).

	+ Default Value: 1e9
	+ Enabled by Default?: true
	+ Datatype: size_t
	+ Syntax: 

	```python
		korali["Solver"]["Termination Criteria"]["Max Resamplings"]["Value"] = *value*
	```

	```python
		korali["Solver"]["Termination Criteria"]["Max Resamplings"]["Enabled"] = *true / false* 
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

	+ Default Value: +std::numeric_limits<float>::max()
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

??? abstract "Min Step Size"

	Specifies the minimal step size of the sample mean from one gneration to another.

	+ Default Value: 1e-12
	+ Enabled by Default?: false
	+ Datatype: double
	+ Syntax: 

	```python
		korali["Solver"]["Termination Criteria"]["Min Step Size"]["Value"] = *value*
	```

	```python
		korali["Solver"]["Termination Criteria"]["Min Step Size"]["Enabled"] = *true / false* 
	```

## Plotting



