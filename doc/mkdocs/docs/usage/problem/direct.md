# Korali / Problem / Direct

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
# Adding two variables, one uniform and another gaussian.

korali["Problem"]["Variables"][0]["Name"] = "X0";
korali["Problem"]["Variables"][0]["Minimum"] = 0.0
korali["Problem"]["Variables"][0]["Maximum"] = 32.0

korali["Problem"]["Variables"][0]["Name"] = "X1";
korali["Problem"]["Variables"][1]["Minimum"] = 0.0
korali["Problem"]["Variables"][1]["Maximum"] = 32.0
```
