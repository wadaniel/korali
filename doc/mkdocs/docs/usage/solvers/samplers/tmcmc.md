# Solvers / TMCMC
   				   
##Description

This is the implementation of the *Transitional Markov Chain Monte Carlo* algorithm, as published in the [paper](https://ascelibrary.org/doi/abs/10.1061/%28ASCE%290733-9399%282007%29133%3A7%28816%29).

TMCMC avoids sampling from difficult target probability densities (e.g. posterior distributions in a Bayesian inference problem) but samples from a series of intermediate PDFs that converge to the target PDF. This technique is also known as Sampling Importance Resampling in the Bayesian community.

**Requirements:**

+ The *Population Size* used at every generation needs to be defined.
+ The *Problem* needs to be of type Bayesian.

##Settings

For a better understanding of the variables please refer to the paper.

```python
  # Definition
  k["Solver"] = "TMCMC";
  
  k["TMCMC"]["Result Output Frequency"] = ...
  
  # Solver Settings
  k["TMCMC"]["Population Size"] = ... 
  k["TMCMC"]["Burn In"] = ...
  k["TMCMC"]["Coefficient of Variation"] = ...
  k["TMCMC"]["Covariance Scaling"] = ...
  k["TMCMC"]["Use Local Covariance"] = ...
  k["TMCMC"]["Min Rho Update"] = ...
  k["TMCMC"]["Max Rho Update"] = ...
  
  # Termination Criteria
  k["TMCMC"]["Termination Criteria"]["Max Generations"]["Active"] = ...
  k["TMCMC"]["Termination Criteria"]["Max Generations"]["Value"] = ...

  # Variable Settings
  k["Variables"][i]["TMCMC"]["Name"] = ...
```


##Solver Settings

- **Result Output Frequency**. Specifies the output frequency of intermediate result files. By default, Korali will set this value to *1* (i.e. each generation). Example:

	```python
    # Reduce the number of result files
	k["TMCMC"]["Result Output Frequency"] = 10

	```

- **Population Size**. Specifies whether the problem evaluation is to be *minimized* or *maximized*. By default, Korali will set this value to *Maximize*. Example:

	```python
	#Maximizing problem evaluation (default)
	k["TMCMC"]["Objective"] = "Maximize"

	#Minimizing problem evaluation
	k["TMCMC"]["Objective"] = "Minimize"
	```
- **Burn In**. Specifies the number of samples $\lambda$ to evaluate per each generation. Example:

	```python
	k["TMCMC"]["Sample Count"] = 32
	```
	
- **Coefficient of Variation**. Controls the learning rate of the conjugate evolution path for $\sigma$ update. By default, Korali will set this value to $\frac{\mu_{eff}+2}{\mu_{eff}+N+3}$. Example:

	```python
	k["TMCMC"]["Sigma Cumulation Factor"] = 1.0
	```

- **Covariance Scaling**. Sets an upper bound for scaling factor $\sigma$. Upper bound is given by the average of the initial standard deviation of the variables. By default, Korali will set this value to *False*. Example:

	```python
	k["TMCMC"]["Sigma Bounded"] = True
	```	
	
- **Use Local Covariance**. Number of best samples to update the covariance matrix and the mean. By default, Korali will set this value to $\lambda / 2$. Example:

	```python
	k["TMCMC"]["Mu"]["Value"] = 8
	```	
	
- **Min Rho Update**. Controls the update of the scaling factor $\sigma$. By default, Korali will precalibrate this value from $\mu_{eff}$ and number of variables (must be larger 0.0). Example:

	```python
	k["TMCMC"]["Damp Factor"] = 1.0
	```	
	
- **Max Rho Update**. Number of resamplings per candidate per generation if sample is outside of Lower and Uppder Bound. By default, Korali will set this value to $10^6$. Example:

	```python
	k["TMCMC"]["Max Resamplings"] = 10e6
	```	

- **Covariance Scaling**. Sets an upper bound for scaling factor $\sigma$. Upper bound is given by the average of the initial standard deviation of the variables. By default, Korali will set this value to *False*. Example:

	```python
	k["TMCMC"]["Sigma Bounded"] = True
	```	
	
- **Use Local Covariance**. Number of best samples to update the covariance matrix and the mean. By default, Korali will set this value to $\lambda / 2$. Example:

	```python
	k["TMCMC"]["Mu"]["Value"] = 8
	```	
		 
## Termination Criteria

- **Max Generations** Specifies the maximum number of generations to run. By default, Korali will set this criterion as active and its value to *1000*. Example:

	```python
	k["TMCMC"]["Termination Criteria"]["Max Generations"]["Active"] = True
	k["TMCMC"]["Termination Criteria"]["Max Generations"]["Value"]  = 2000
	```

## Variable Settings

- **Name** Specifies the lower bound for the variable's value. Korali will not generate samples in which this variable falls below the specified minimum. By default, Korali will set this value to *-Infinity*. Example:

	```python
	# Modifying the lower bound of two variables
	k["Variables"][i]["TMCMC"][0]["Minimum"] = 0.0;
	k["Variables"][i]["TMCMC"][1]["Minimum"] = -32.0;
	```

## Plotting

Here we explain technical details of the **TMCMC** result plot.

The `python3 -m korali.plotter` command plots the distribution of the samples at every
generation. The samples are read from the json-files stored in the output
directory (`/_korali_result/`).

A plot of the samples obtained after the final gerneration of TMCMC
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
of samples that can be found in a given area.
