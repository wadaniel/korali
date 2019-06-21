# Problems / Bayesian / Reference Likelihood

##Description

In a *Bayesian* problem, the defines and a prior probability density $p(\vartheta)$ for the problem variables, and the solver is applied to the of the posterior distribution:

 $$ p(\vartheta | d) = \frac{p(d | \vartheta) p(\vartheta)}{p(d)} $$

Whereas with a *Reference Likelihood*, the computational model is assumed to be of the form $f(x;\vartheta)$, where $d$ is a set of *M* given data points. The output of the model represents the values of the function at the given points for which Korali can build a likelihood function $p(d|\vartheta)$, and a prior probability density $p(\vartheta)$. 

Currently, Korali uses a Gaussian estimator for the error component of the likelihood calculation, using a statistical-type variable, *sigma*:

$$ p(d | \vartheta) = {\frac {1}{\sigma {\sqrt {2\pi }}}}e^{-{\frac {1}{2}}\left((x-\mu )/\sigma \right)^{2}} $$

**Requirements:**

+ At least one computational variable as input for the computational model.
+ At least one statistical variable for the value of *sigma* in the Gaussian error calculation.
+ A [Reference](/usage/models/reference) computational model should be defined for the likelihood, and should return exactly *M* results.
+ A prior distribution should be defined for every variable.


##Syntax

```python
  # Definition
  k["Problem"] = "Bayesian"
	
  # Problem Settings
  k["Bayesian"]["Likelihood"]["Type"] = "Reference"
  k["Bayesian"]["Likelihood"]["Reference Data"][i] = ...
	
  # Variable Settings
  k["Variables"][i]["Bayesian"]["Type"] = ...
  k["Variables"][i]["Bayesian"]["Prior Distribution"] ...
```

##Problem Settings

- **Reference Data**. The reference data are points in the variable space that the computational model model evaluates, given the sample variable data. The output of the model allows Korali to evaluate the likelihood function $p(d|\vartheta)$

	Example:
	```python
	# Adding reference data items from a vector with the += operator.
	for x in refData:
	 korali["Problem"]["Reference Data"] += x
	 
	# Adding reference data items from a vector by index.
	for i in range(len(refData)):
	 korali["Problem"]["Reference Data"][i] = refData[i]
	 
	# Providing a complete vector as reference data
	 korali["Problem"]["Reference Data"] = refData
	```

##Variable Settings

- **Type**. The Bayesian problem makes a distinction between computational and statistical variable types. Computational variables describe the dimension of the problem-space (e.g., the X, Y, and Z coordinates of a real-world 3D problem), while statistical variables are employed to infer values from the statistical model (e.g., the error estimation $\sigma$ of a Gaussian process).

	Example:
	
	```python
	k["Variables"][0]["Bayesian"]["Type] = "Computational"
	k["Variables"][1]["Bayesian"]["Type] = "Statistical"
	```

- **Prior Distribution**. The Bayesian problem type extends the definition of each variable with the possibility of defining a prior distribution. Korali currently offers the following distributions:

	- [Cauchy](../../distributions/#cauchy)
	- [Exponential](../../distributions/#exponential)
	- [Gamma](../../distributions/#gamma)
	- [Gaussian](../../distributions/#gaussian)
	- [Laplace](../../distributions/#laplace)
	- [Uniform](../../distributions/#uniform)

	Example:
	
	```python
	  #Setting a uniform prior distribution for variable 0
	  k["Variables"][0]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform"
	  k["Variables"][0]["Bayesian"]["Prior Distribution"]["Minimum"] = -10.0
	  k["Variables"][0]["Bayesian"]["Prior Distribution"]["Maximum"] = +10.0
	```
