# Evaluation Type / Bayesian

##Description

In a Bayesian problem type, the computational model is assumed to be of the form $f(x;\vartheta)$, where $d$ is a set of given data points. The output of the model represents the values of the function at the given points for which Korali can build a likelihood function $p(d|\vartheta)$, and a prior probability density $p(\vartheta)$. The solver method will then optimize or sample (depending on the user's choice) the problem's variable space given the value of the posterior distribution:

 $$ p(\vartheta | d) = \frac{p(d | \vartheta) p(\vartheta)}{p(d)} $$

##Syntax
       korali["Problem"]["Type"] = "Bayesian"

##Requirements

+ It should contain at least one computational variable as input for the computational model.
+ It should contain at least one statistical variable for the value of sigma in the Gaussian error calculation.

##Settings

+ Name: **Reference Data**
     - Description: Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
     - Type: Array of Doubles
	 - Default: *{empty}*
	 - Requirements: The number of reference data items should be equal to the number of results returned by the computational model.
	 - Syntax: `korali["Problem"]["Reference Data"][i] = myValue;`
	 - Example:

```python
# Adding reference data items from a vector.

for x in refData:
 korali["Problem"]["Reference Data"] += x
```
