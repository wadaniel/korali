# Problems / Bayesian / Direct Likelihood

##Description

In a *Bayesian* problem, the defines and a prior probability density $p(\vartheta)$ for the problem variables, and the solver is applied to the of the posterior distribution:

 $$ p(\vartheta | d) = \frac{p(d | \vartheta) p(\vartheta)}{p(d)} $$

Whereas with a *Direct Likelihood*, the function $p(d|\vartheta)$ is given directly by a user-defined model of the form $f:\mathbb{R}^N\rightarrow\mathbb{R}$, where $N$ is the number of variables.

** Usage **

	```python
	  k["Problem"] = "Bayesian"
	  k["Bayesian"]["Likelihood"]["Type"] = "Direct"
	```
	

**Requirements:**

+ A [Simple](/usage/models/simple) computational model should be defined for the likelihood.
+ At least one variable should be defined.
+ A prior distribution should be defined for every variable.

##Problem Settings

??? abstract "Likelihood Type"
	
	Defines how the likelihood is calculated. In the Direct Likelihood type, it is provided directly by the computational model.

	Example:
	```python
	  #Setting direct likelihood evaluation.
	  k["Bayesian"]["Likelihood"]["Type"] = "Direct"
	```
	
##Variable Settings

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