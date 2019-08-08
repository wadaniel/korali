# Transitional Markov Chain Monte Carlo 

## Description

This is the implementation of the *Transitional Markov Chain Monte Carlo*
algorithm, as published in [Ching2007](https://ascelibrary.org/doi/abs/10.1061/%28ASCE%290733-9399%282007%29133%3A7%28816%29).

TMCMC avoids sampling from difficult target probability densities
(e.g. posterior distributions in a Bayesian inference problem) but samples from
a series of intermediate PDFs that converge to the target PDF.
This technique is also known as Sampling Importance Resampling in the Bayesian community.

** Usage **

```python
  k["Solver"] = "TMCMC"
```
	
**Requirements:**

+ The *Population Size* used at every generation needs to be defined.
+ The *Problem* needs to be of type Bayesian.


## Solver Settings

??? abstract "Population Size"

	Specifies the number of samples drawn from the posterior distribution at each generation.

	+ Default Value: 
	+ Datatype: size_t
	+ Syntax: 

	```python
		korali["Solver"]["Population Size"] = *value*
	```

??? abstract "Burn In Default"

	Specifies the number of additional TMCMC steps per sample per generation.
	Note that only the last sample per chain is considered for the recombination.

	+ Default Value: 0
	+ Datatype: size_t
	+ Syntax: 

	```python
		korali["Solver"]["Burn In Default"] = *value*
	```

??? abstract "Burn In Steps"

	Specifies the number of additional TMCMC steps per sample at specified
	generations.

	+ Default Value: 0
	+ Datatype: std::vector<size_t>
	+ Syntax: 

	```python
		korali["Solver"]["Burn In Steps"] = *value*
	```

??? abstract "Target Coefficient of Variation"

	Initial value for the target coefficient of variation to search for the exponent
	 $\rho_{i+1}$. By default, Korali will set this value to 1.00 as suggested in 
	 [Ching2007].

	+ Default Value: 1.0
	+ Datatype: double
	+ Syntax: 

	```python
		korali["Solver"]["Target Coefficient of Variation"] = *value*
	```

??? abstract "Covariance Scaling"

	Covariance scaling factor $\beta^2$ of proposal distribution.
	By default, Korali sets this value 0.04 as suggested in [Ching2007].

	+ Default Value: 0.04
	+ Datatype: double
	+ Syntax: 

	```python
		korali["Solver"]["Covariance Scaling"] = *value*
	```

??? abstract "Use Local Covariance"

	If $true, TMCMC calculates a local covariance matrix per sample from its 
	neighbours.

	+ Default Value: false
	+ Datatype: bool
	+ Syntax: 

	```python
		korali["Solver"]["Use Local Covariance"] = *value*
	```

??? abstract "Min Rho Update"

	Minimum increment of the exponent $\rho_{i+1}$. This parameter prevents TMCMC 
	from stalling.

	+ Default Value: 0.00001
	+ Datatype: double
	+ Syntax: 

	```python
		korali["Solver"]["Min Rho Update"] = *value*
	```

??? abstract "Max Rho Update"

	Maximum increment of the exponent $\rho{i+1}$. By default, Korali will set this 
	value to 1.0 (inactive).

	+ Default Value: 1.0
	+ Datatype: double
	+ Syntax: 

	```python
		korali["Solver"]["Max Rho Update"] = *value*
	```

## Variable Settings

## Termination Criteria

## Plotting

Here we explain technical details of the **TMCMC** result plot.

The `python3 -m korali.plotter` command plots the distribution of the samples at every
generation. The samples are read from the json-files stored in the output
directory (`/_korali_result/`).

A plot of the samples obtained after the final generation of TMCMC
function is given below. Here, the target function is the exponential of the
negative of the 2-dimensional [Rosenbrock](https://en.wikipedia.org/wiki/Rosenbrock_function)
function.

![figure](sampling_rosenbrock.png)

**Diagonal Plots**. The diagonal plots show the marginal probability densities of the samples for
each variable. Note that the indices of the vertical axes correspond to the
upper and lower triangle plot and not to the diagonal plots.

**Upper Triangle**. In the plots in the upper triangle we see the actual realization of the samples.
The axes represent the dimensions, respectively the objective variables,
of the problem and we show a two-dimensional plot for every variable pair.

**Lower Triangle**. The lower triangle shows the probability density of the samples for each variable pair.
The density is approximated through a smoothening operator applied to the number
of samples that can be found in a given area.


