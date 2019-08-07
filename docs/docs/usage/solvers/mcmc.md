# Markov Chain Monte Carlo 

## Description

This is an implementation of the *Delayed Rejection Adaptive Metropolis* algorithm,
as published in [Haario2006](https://link.springer.com/article/10.1007%2Fs11222-006-9438-0).
This solver can also be configured to run the standard *Metropolis Hastings* method.

** Usage **

```python
  k["Solver"] = "MCMC"
```
	
**Requirements:**

+ The *Chain Length*, i.e. the number of generated samples.
+ The *Initial Mean* of the proposal distribution for each variable.
+ The *Standard Deviation* of the proposal distribution for each variable.


## Solver Settings

??? abstract "Max Chain Length"

	Number of samples to be generated.

	+ Default Value: 
	+ Datatype: size_t
	+ Syntax: 

	```python
		korali["Solver"]["Max Chain Length"] = *value*
	```

??? abstract "Burn In"

	Specifies the number of preliminary MCMC steps before samples are being drawn.
	This may reduce effects from improper initialization.

	+ Default Value: 0
	+ Datatype: size_t
	+ Syntax: 

	```python
		korali["Solver"]["Burn In"] = *value*
	```

??? abstract "Rejection Levels"

	Controls the number of accept-reject stages per MCMC step.

	+ Default Value: 1
	+ Datatype: size_t
	+ Syntax: 

	```python
		korali["Solver"]["Rejection Levels"] = *value*
	```

??? abstract "Use Adaptive Sampling"

	Specifies if covariance matrix of the proposal distribution is calculated from
	the samples.

	+ Default Value: false
	+ Datatype: bool
	+ Syntax: 

	```python
		korali["Solver"]["Use Adaptive Sampling"] = *value*
	```

??? abstract "Non Adaption Period"

	Number of steps (after Burn In steps) during which the initial standard deviation
	is used (only relevant for Adaptive Sampling). If "0" is specified, Korali will
	set this value to $5%$ of the Chain Length.

	+ Default Value: 0
	+ Datatype: size_t
	+ Syntax: 

	```python
		korali["Solver"]["Non Adaption Period"] = *value*
	```

??? abstract "Chain Covariance Scaling"

	Multiplier inside of Chain Covariance update in order to control the covariance.

	+ Default Value: 1.0
	+ Datatype: double
	+ Syntax: 

	```python
		korali["Solver"]["Chain Covariance Scaling"] = *value*
	```

## Variable Settings

??? abstract "Initial Mean"

	Specifies the Initial Mean of the proposal distribution.

	+ Default Value: 
	+ Datatype: double
	+ Syntax: 

	```python
		korali["Variables"][i]["Initial Mean"] = *value*
	```

??? abstract "Initial Standard Deviation"

	Specifies the Standard Deviation for each variable. The proposal distribution
	is defined through a covariance matrix with the variance of the variables in
	its diagonal.

	+ Default Value: 
	+ Datatype: double
	+ Syntax: 

	```python
		korali["Variables"][i]["Initial Standard Deviation"] = *value*
	```

## Termination Criteria

## Plotting

Here we explain technical details of the **MCMC** result plot.

The `python3 -m korali.plotter` command plots the distribution of the samples at every
generation. The samples are read from the json-files stored in the output
directory (`/_korali_result/`).

A plot of the samples obtained after the final gerneration of MCMC
function is given below. Here, the target function is the exponential of the
negative of the 2-dimensional [Rosenbrock](https://en.wikipedia.org/wiki/Rosenbrock_function)
function.

![figure](sampling_rosenbrock.png)

**Diagonal Plots**. The diagonal plots show the marginal probability densities of the samples for
each variable. Note that the indices of the vertical axes correspond to the
upper and lower triangle plot and not to the diagonal plots.

**Upper Triangle**. In the plots in the upper triangle we see the actual realization of the samples.
The axes represent the dimensions, respectiely the objective variables,
of the problem and we show a two-dimensional plot for every variable pair.

**Lower Triangle**. The lower triangle shows the probability density of the samples for each variable pair.
The density is approximated through a smoothening operator applied to the number


