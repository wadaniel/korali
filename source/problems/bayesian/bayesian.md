# Bayesian Inference

##Description

In a *Bayesian Inference* problem, the defines and a prior probability density $p(\vartheta)$ for the problem variables, and the solver is applied to the of the posterior distribution:

 $$ p(\vartheta | d) = \frac{p(d | \vartheta) p(\vartheta)}{p(d)} $$

## Likelihood Models 

### Additive Gaussian Likelihood

Whereas with a *Additive Gaussian Likelihood*, the computational model is assumed to be of the form $f(x;\vartheta)$, where $d$ is a set of *M* given data points. The output of the model represents the values of the function at the given points for which Korali can build a likelihood function $p(d|\vartheta)$, and a prior probability density $p(\vartheta)$. 

Currently, Korali uses a Gaussian estimator for the error component of the likelihood calculation, using a statistical-type variable, *sigma*:

$$ p(d | \vartheta) = {\frac {1}{\sigma {\sqrt {2\pi }}}}e^{-{\frac {1}{2}}\left((x-\mu )/\sigma \right)^{2}} $$

** Usage **

```python
  k["Problem"] = "Bayesian Inference"
  k["Bayesian"]["Likelihood"]["Type"] = "Reference"
  k["Bayesian"]["Likelihood"]["Reference Data"] = myRefDataArray
```
	
**Requirements:**

+ At least one computational variable as input for the computational model.
+ At least one statistical variable for the value of *sigma* in the Gaussian error calculation.
+ A [Reference](/usage/models/reference) computational model should be defined for the likelihood, and should return exactly *M* results.
+ A prior distribution should be defined for every variable.

### Custom Likelihood

With a *Custom Likelihood*, the function $p(d|\vartheta)$ is given directly by a user-defined model of the form $f:\mathbb{R}^N\rightarrow\mathbb{R}$, where $N$ is the number of variables.

** Usage **

```python
  k["Problem"] = "Bayesian Inference"
  k["Problem"]["Likelihood"]["Model"] = "Custom"
```
	
**Requirements:**

+ A [Simple](/usage/models/simple) computational model should be defined for the likelihood.
+ At least one variable should be defined.
+ A prior distribution should be defined for every variable.
	
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
	  k["Variables"][0]["Prior Distribution"]["Type"] = "Uniform"
	  k["Variables"][0]["Prior Distribution"]["Minimum"] = -10.0
	  k["Variables"][0]["Prior Distribution"]["Maximum"] = +10.0
	```
	
??? abstract "Type" 

	The Bayesian problem makes a distinction between computational and statistical variable types. Computational variables describe the dimension of the problem-space (e.g., the X, Y, and Z coordinates of a real-world 3D problem), while statistical variables are employed to infer values from the statistical model (e.g., the error estimation $\sigma$ of a Gaussian process).

	Example:
	
	```python
	k["Variables"][0]["Type] = "Computational"
	k["Variables"][1]["Type] = "Statistical"
	```