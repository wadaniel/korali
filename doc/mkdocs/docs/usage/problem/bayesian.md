# Korali / Problem / Bayesian

##Description

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

##Syntax
       korali["Problem"]["Type"] = "Bayesian"

##Requirements

+ It should contain at least one computational variable as input for the computational model.
+ It should contain at least one statistical variable for the value of sigma in the Gaussian error calculation.

##Settings

+ Name: **Variables**
     - Description: Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
     - Type: Array of [Variables](../variables/uniform.md)
	 - Default: *{empty}*
	 - Syntax: `korali["Problem"]["Variables"][i]`, where i is the variable index and the syntax for [Variables](../variables/uniform.md) follows.
	 - Example:

```python
# Adding two variables, one Computational Gaussian and one Statistical Uniform.

korali["Problem"]["Variables"][0]["Name"] = "X";
korali["Problem"]["Variables"][0]["Type"] = "Computational";
korali["Problem"]["Variables"][0]["Distribution"] = "Gaussian";
korali["Problem"]["Variables"][0]["Mean"]  = 0.0;
korali["Problem"]["Variables"][0]["Sigma"] = 1.0;

korali["Problem"]["Variables"][1]["Name"] = "Sigma";
korali["Problem"]["Variables"][1]["Type"] = "Statistical";
korali["Problem"]["Variables"][1]["Distribution"] = "Uniform";
korali["Problem"]["Variables"][1]["Minimum"]  = 0.0
korali["Problem"]["Variables"][1]["Maximum"] = 5.0


```

<br>

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
