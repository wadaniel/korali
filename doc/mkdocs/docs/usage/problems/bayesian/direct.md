# Problems / Bayesian / Direct Likelihood

##Description

In a *Bayesian* problem, the defines and a prior probability density $p(\vartheta)$ for the problem variables, and the solver is applied to the of the posterior distribution:

 $$ p(\vartheta | d) = \frac{p(d | \vartheta) p(\vartheta)}{p(d)} $$

Whereas with a *Direct Likelihood*, the function $p(d|\vartheta)$ is given directly by a user-defined model of the form $f:\mathbb{R}^N\rightarrow\mathbb{R}$, where $N$ is the number of variables.

**Requirements:**

+ A [Simple](/usage/models/simple) computational model should be defined for the likelihood.
+ At least one variable should be defined.
+ A prior distribution should be defined for every variable.

##Syntax
```python
  # Definition
  k["Problem"] = "Bayesian"
  
  # Problem Settings
  k["Bayesian"]["Likelihood"]["Type"] = "Direct"
	
  # Variable Settings
  k["Variables"][i]["Bayesian"]["Prior Distribution"] ...
```

##Problem Settings

- **Likelihood Type**. Defines how the likelihood is calculated. In the Direct Likelihood type, it is provided directly by the computational model.

	Example:
	```python
	  #Setting direct likelihood evaluation.
	  k["Bayesian"]["Likelihood"]["Type"] = "Direct"
	```
	
##Variable Settings

- **Prior Distribution**. The Bayesian problem type extends the definition of each variable with the possibility of defining a prior distribution. Korali currently offers the following distributions:

	- [Cauchy](/usage/distributions/cauchy)
	- [Exponential](/usage/distributions/exponential)
	- [Gamma](/usage/distributions/gamma)
	- [Gaussian](/usage/distributions/gaussian)
	- [Laplace](/usage/distributions/laplace)
	- [Uniform](/usage/distributions/uniform)

	Example:
	```python
	  #Setting a uniform prior distribution for variable 0
	  k["Variables"][0]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform"
	  k["Variables"][0]["Bayesian"]["Prior Distribution"]["Minimum"] = -10.0
	  k["Variables"][0]["Bayesian"]["Prior Distribution"]["Maximum"] = +10.0
	```