# Solvers / MCMC
   				   
##Description

This is am implementation of the *Delayed Rejection Adaptive Metropolis* algorithm, as published in this [article](https://link.springer.com/article/10.1007%2Fs11222-006-9438-0). We can also run standard *Metropolis Hastings* if correctly configured.


**Requirements:**

+ The *Chain Length*, i.e. the number of generated samples.
+ The *Problem* needs to be of type Bayesian. (TODO: confirm)

##Settings

For a better understanding of the variables please refer to the paper.

```python
  # Definition
  k["Solver"] = "MCMC";
  
  k["MCMC"]["Result Output Frequency"] = ...
  
  # Solver Settings
  k["MCMC"]["Population Size"] = ... 
  k["MCMC"]["Burn In"] = ...
  k["MCMC"]["Rejection Levels"] = ...
  k["MCMC"]["Adaptive Sampling"] = ...
  k["MCMC"]["Non Adaption Period"] = ...
  k["MCMC"]["Chain Covariance Learning Rate"] = ...
  k["MCMC"]["Chain Covariance Increment"] = ...
  k["MCMC"]["Max Resamplings"] = ...
  
  # Termination Criteria
  k["MCMC"]["Termination Criteria"]["Max Function Evaluations"]["Active"] = ...
  k["MCMC"]["Termination Criteria"]["Max Function Evaluations"]["Value"] = ...
  k["MCMC"]["Termination Criteria"]["Max Candidate Proposals"]["Active"] = ...
  k["MCMC"]["Termination Criteria"]["Max Candidate Proposals"]["Value"] = ...

  # Variable Settings
  k["Variables"][i]["MCMC"]["Name"] = ...
  k["Variables"][i]["MCMC"]["Initial Mean"] = ...
  k["Variables"][i]["MCMC"]["Initial Standard Deviation"] = ...
```


##Solver Settings

- **Result Output Frequency**. Specifies the output frequency of intermediate result files. By default, Korali will set this value to *1* (i.e. each generation). Example:

	```python
    # Reduce the number of result files
	k["MCMC"]["Result Output Frequency"] = 10

	```

- **Population Size**. Specifies whether the problem evaluation is to be *minimized* or *maximized*. By default, Korali will set this value to *Maximize*. Example:

	```python
	#Maximizing problem evaluation (default)
	k["MCMC"]["Objective"] = "Maximize"

	#Minimizing problem evaluation
	k["MCMC"]["Objective"] = "Minimize"
	```
- **Burn In**. Specifies the number of samples $\lambda$ to evaluate per each generation. Example:

	```python
	k["MCMC"]["Sample Count"] = 32
	```
	
- **Rejection Levels**. Controls the learning rate of the conjugate evolution path for $\sigma$ update. By default, Korali will set this value to $\frac{\mu_{eff}+2}{\mu_{eff}+N+3}$. Example:

	```python
	k["MCMC"]["Sigma Cumulation Factor"] = 1.0
	```

- **Adaptive Sampling**. Sets an upper bound for scaling factor $\sigma$. Upper bound is given by the average of the initial standard deviation of the variables. By default, Korali will set this value to *False*. Example:

	```python
	k["MCMC"]["Sigma Bounded"] = True
	```	
	
- **Non Adaption Period**. Number of best samples to update the covariance matrix and the mean. By default, Korali will set this value to $\lambda / 2$. Example:

	```python
	k["MCMC"]["Mu"]["Value"] = 8
	```	
	
- **Chain Covariance Learning Rate**. Controls the update of the scaling factor $\sigma$. By default, Korali will precalibrate this value from $\mu_{eff}$ and number of variables (must be larger 0.0). Example:

	```python
	k["MCMC"]["Damp Factor"] = 1.0
	```	
	
- **Chain Covariance Increment**. Number of resamplings per candidate per generation if sample is outside of Lower and Uppder Bound. By default, Korali will set this value to $10^6$. Example:

	```python
	k["MCMC"]["Max Resamplings"] = 10e6
	```	

- **Max Resamplings**. Sets an upper bound for scaling factor $\sigma$. Upper bound is given by the average of the initial standard deviation of the variables. By default, Korali will set this value to *False*. Example:

	```python
	k["MCMC"]["Sigma Bounded"] = True
	```	
	
## Termination Criteria

- **Max Generations** Specifies the maximum number of generations to run. By default, Korali will set this criterion as active and its value to *1000*. Example:

	```python
	k["MCMC"]["Termination Criteria"]["Max Generations"]["Active"] = True
	k["MCMC"]["Termination Criteria"]["Max Generations"]["Value"]  = 2000
	```

- **Max Candidate Proposals** Specifies the maximum number of generations to run. By default, Korali will set this criterion as active and its value to *1000*. Example:

	```python
	k["MCMC"]["Termination Criteria"]["Max Generations"]["Active"] = True
	k["MCMC"]["Termination Criteria"]["Max Generations"]["Value"]  = 2000
	```

## Variable Settings

- **Name** Specifies the lower bound for the variable's value. Korali will not generate samples in which this variable falls below the specified minimum. By default, Korali will set this value to *-Infinity*. Example:

	```python
	# Modifying the lower bound of two variables
	k["Variables"][i]["MCMC"][0]["Minimum"] = 0.0;
	k["Variables"][i]["MCMC"][1]["Minimum"] = -32.0;
	```

- **Initial Mean** Specifies the lower bound for the variable's value. Korali will not generate samples in which this variable falls below the specified minimum. By default, Korali will set this value to *-Infinity*. Example:

	```python
	# Modifying the lower bound of two variables
	k["Variables"][i]["MCMC"][0]["Minimum"] = 0.0;
	k["Variables"][i]["MCMC"][1]["Minimum"] = -32.0;
	```

- **Initial Standard Deviation** Specifies the lower bound for the variable's value. Korali will not generate samples in which this variable falls below the specified minimum. By default, Korali will set this value to *-Infinity*. Example:

	```python
	# Modifying the lower bound of two variables
	k["Variables"][i]["MCMC"][0]["Minimum"] = 0.0;
	k["Variables"][i]["MCMC"][1]["Minimum"] = -32.0;
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
