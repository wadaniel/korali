# Korali / Problem / Bayesian

##Description

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

##Syntax
       Korali["Problem"]["Type"] = "Bayesian"

##Requirements

+ It should contain at least one computational variable as input for the computational model.
+ It should contain at least one statistical variable for the value of sigma in the Gaussian error calculation.

##Settings

+ Name: **Variables**
     - Description: Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
     - Type: Array of [Variables](../variables/uniform.md)
	 - Default: *{empty}*
	 - Syntax: `Korali["Problem"]["Variables"][i]`, where i is the variable index and the syntax for [Variables](../variables/uniform.md) follows.
	 - Example:

```python
# Adding two variables, one Computational Gaussian and one Statistical Uniform.

Korali["Problem"]["Variables"][0]["Name"] = "X";
Korali["Problem"]["Variables"][0]["Type"] = "Computational";
Korali["Problem"]["Variables"][0]["Distribution"] = "Gaussian";
Korali["Problem"]["Variables"][0]["Mean"]  = 0.0;
Korali["Problem"]["Variables"][0]["Sigma"] = 1.0;

Korali["Problem"]["Variables"][1]["Name"] = "Sigma";
Korali["Problem"]["Variables"][1]["Type"] = "Statistical";
Korali["Problem"]["Variables"][1]["Distribution"] = "Uniform";
Korali["Problem"]["Variables"][1]["Minimum"]  = 0.0
Korali["Problem"]["Variables"][1]["Maximum"] = 5.0


```

<br>

+ Name: **Reference Data**
     - Description: Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
     - Type: Array of Doubles
	 - Default: *{empty}*
	 - Requirements: The number of reference data items should be equal to the number of results returned by the computational model.
	 - Syntax: `Korali["Problem"]["Reference Data"][i] = myValue;`
	 - Example:

```python
# Adding reference data items from a vector.

for x in refData:
 Korali["Problem"]["Reference Data"] += x
```
