# Evaluation Type / Bayesian

##Description

In a Bayesian problem type, the computational model is assumed to be of the form $f(x;\vartheta)$, where $d$ is a set of given data points. The output of the model represents the values of the function at the given points for which Korali can build a likelihood function $p(d|\vartheta)$, and a prior probability density $p(\vartheta)$. The solver method will then optimize or sample (depending on the user's choice) the problem's variable space given the value of the posterior distribution:

 $$ p(\vartheta | d) = \frac{p(d | \vartheta) p(\vartheta)}{p(d)} $$

##Syntax
```python
  # Definition
  korali["Problem"]["Type"] = "Bayesian"
  
  # Mandatory Settings
  korali["Problem"]["Reference Data"][i] = ...
```

##Requirements

+ It should contain at least one computational variable as input for the computational model.
+ It should contain at least one statistical variable for the value of sigma in the Gaussian error calculation.
+ The computational model should return *n* results, where *n* is the number of reference data items.

##Settings

+ Name: **Reference Data**
     - Description: The reference data are points in the variable space that the computational model model evaluates, given the sample variable data. The output of the model allows Korali to evaluate the likelihood function $p(d|\vartheta)$
	 - Default Value: *None*.
	 - Examples:

```python
# Adding reference data items from a vector with the += operator.
for x in refData:
 korali["Problem"]["Reference Data"] += x
 
# Adding reference data items from a vector by index.
for i in range(len(refData)):
 korali["Problem"]["Reference Data"][i] = refData[i]
```
