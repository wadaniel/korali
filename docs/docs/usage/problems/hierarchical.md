# Hierarchical Bayesian Modeling

Hierarchical Bayesian description


## Configuration

### Problem Settings
	
### Variable-Specific Settings

??? abstract "Prior Distribution"

	The Bayesian problem type extends the definition of each variable with the possibility of defining a prior distribution. Korali currently offers the following distributions:

	- [Cauchy](../../../distributions/#cauchy)
	- [Exponential](../../../distributions/#exponential)
	- [Gamma](../../../distributions/#gamma)
	- [Gaussian](../../../distributions/#gaussian)
	- [Laplace](../../../distributions/#laplace)
	- [Uniform](../../../distributions/#uniform)

	Example:
	```python
	  #Setting a uniform prior distribution for variable 0
	  k["Variables"][0]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform"
	  k["Variables"][0]["Bayesian"]["Prior Distribution"]["Minimum"] = -10.0
	  k["Variables"][0]["Bayesian"]["Prior Distribution"]["Maximum"] = +10.0
	```
	
??? abstract "Type" 

	The Bayesian problem makes a distinction between computational and statistical variable types. Computational variables describe the dimension of the problem-space (e.g., the X, Y, and Z coordinates of a real-world 3D problem), while statistical variables are employed to infer values from the statistical model (e.g., the error estimation $\sigma$ of a Gaussian process).

	Example:
	
	```python
	k["Variables"][0]["Bayesian"]["Type] = "Computational"
	k["Variables"][1]["Bayesian"]["Type] = "Statistical"
	```