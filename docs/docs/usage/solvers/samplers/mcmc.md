# Solvers / MCMC
   				   
##Description

This is am implementation of the *Delayed Rejection Adaptive Metropolis* algorithm, as published in [Haario2006](https://link.springer.com/article/10.1007%2Fs11222-006-9438-0). We can also run standard *Metropolis Hastings* if correspondingly configured.

**Requirements:**

+ The *Chain Length*, i.e. the number of generated samples.
+ The *Initial Mean* of the proposal distribution for each variable.
+ The *Standard Deviation* of the proposal distribution for each variable.

##Settings

??? abstract "Result Output Frequency"

	Specifies the output frequency of intermediate result files. By default, Korali sets this value to *100*. Example:

	```python
    # Increase the number of result files
	k["MCMC"]["Result Output Frequency"] = 25

	```

??? abstract "Chain Length"

	Specifies the number of samples drawn from the objective function. Example:

	```python
	# Specifying Chain Length
	k["MCMC"]["Chain Length"] = 5000
	```

??? abstract "Burn In"

	Specifies the number of prelimnieary MCMC steps before samples are being drawn. This may reduce effects from improper initialization. By default, Korali sets this value *1*. Example:

	```python
    # Add Burn In steps
	k["MCMC"]["Burn In"] = 500
	```
	
??? abstract "Rejection Levels"

	Controls the number of accept-reject stages per MCMC step. By default, Korali sets this value *1* (standard Metropolis Hastings). Example:

	```python
    # Introduce Delayed Rejection
	k["MCMC"]["Rejection Levels"] = 2
	```

??? abstract "Adaptive Sampling"

	Specifies if covariance matrix of the proposal distribution is calculated from the samples. By default, Korali sets this flag *False* (standard Metropolis Hastings). Example:

	```python
    # Intoduce Adaptive Sampling
	k["MCMC"]["Adaptive Sampling"] = True
	```	
	
??? abstract "Non Adaption Period"

	Number of steps (after Burn In steps) during which the initial standard deviation is used (only relevant for Adaptive Sampling). By default, Korali sets this value to $5%$ of the Chain Length. Example:

	```python
    # Reducing Non Adaption Period
	k["MCMC"]["Non Adaption Period"] = 100
	```	
	
??? abstract "Chain Covariance Scaling"

	Scaling factor of the chain covariance (only relevant for Adaptive Sampling). By default, Korali will precalibrate this factor from the number of variables: $\frac{2.4^2}{N}$ [Gelman1995]. Example:

	```python
    # Increase Scaling
	k["MCMC"]["Chain Covariance Scaling"] = 4.0
	```	
	
??? abstract "Chain Covariance Increment"

	Small constant to avoid singularity of the chain covariance. By default, Korali will set this value to $10^{-3}$. Example:

	```python
    # Setting Chain Covariance Increment
	k["MCMC"]["Max Resamplings"] = 10e-5
	```	

## Termination Criteria

??? abstract "Max Generations"

	Specifies the maximum number of function evaluations (only relevant if Rejection Levels is large). By default, Korali will set this criterion as inactive. Example:

	```python
    # Set an upper bound for number of function evaluations
	k["MCMC"]["Termination Criteria"]["Max Function Evaluations"]["Active"] = True
	k["MCMC"]["Termination Criteria"]["Max Function Evaluations"]["Value"]  = 1e6
	```


## Variable Settings

??? abstract "Initial Mean"
	
	Specifies the Initial Mean of the proposal distribution. Example:

	```python
	# Specifying the Initial Mean
	k["Variables"][0]["MCMC"]["Intiial Mean"] = 1.0;
	k["Variables"][1]["MCMC"]["Initial Mean"] = 1.0;
	```

??? abstract "Standard Deviation"

	Specifies the Standard Deviation for each variable. The proposal distribution is defined through a covariance matrix with the variance of the variables in its diagonal. Example:

	```python
	# Specifying the Standard Deviation
	k["Variables"][i]["MCMC"][0]["Standard Deviation"] = 2.0;
	k["Variables"][i]["MCMC"][1]["Standard Deviation"] = 2.0;
	```

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
o
