# Korali / Problem / Direct

!!! warning
    This page serves as a placeholder and is under construction.

##Description

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

##Syntax
       korali["Problem"]["Type"] = "Direct"

##Requirements

+ It should contain at least one computational variable as input for the computational model.

##Settings

+ Name: **Variables**
     - Description: Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
     - Type: Array of [Variables](../variables/uniform.md)
	 - Default: *{empty}*
	 - Syntax: `korali["Problem"]["Variables"][i]`, where i is the variable index and the syntax for [Variables](../variables/uniform.md) follows.
	 - Example:

```python
# Adding two variables, one uniform and one gaussian.

korali["Problem"]["Variables"][1]["Name"] = "X0";
korali["Problem"]["Variables"][1]["Type"] = "Computational";
korali["Problem"]["Variables"][1]["Distribution"] = "Uniform";
korali["Problem"]["Variables"][1]["Minimum"]  = 0.0
korali["Problem"]["Variables"][1]["Maximum"] = 5.0

korali["Problem"]["Variables"][0]["Name"] = "X1";
korali["Problem"]["Variables"][0]["Type"] = "Computational";
korali["Problem"]["Variables"][0]["Distribution"] = "Gaussian";
korali["Problem"]["Variables"][0]["Mean"]  = 0.0;
korali["Problem"]["Variables"][0]["Sigma"] = 1.0;
```
