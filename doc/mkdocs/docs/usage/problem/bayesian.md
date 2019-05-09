# Korali / Problem / Bayesian
				   
##Description


	   
##Syntax
       korali["Problem"]["Type"] = "Bayesian"


##Settings

+ Name: **Variables**
     - Description: Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
     - Type: Array of [Variables](../variables/uniform.md)
	 - Default: {empty}
	 - Requirements: At least one variable should be added to the problem. 
	 - Syntax: `korali["Problem"]["Variables"][i]`, where i is the variable index and the syntax for [Variables](../variables/uniform.md) follows.
	 - Example:
	 
```python
# Adding two variables, one uniform and another gaussian.

korali["Problem"]["Variables"][0]["Distribution"] = "Uniform"
korali["Problem"]["Variables"][0]["Minimum"] = 0.0
korali["Problem"]["Variables"][0]["Maximum"] = 32.0

korali["Problem"]["Variables"][1]["Distribution"] = "Gaussian"
korali["Problem"]["Variables"][1]["Minimum"] = 0.0
korali["Problem"]["Variables"][1]["Maximum"] = 32.0
```

<br>
	 
+ Name: **Reference Data**
     - Description: Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
     - Type: Array of Doubles
	 - Default: {empty}
	 - Requirements: The number of reference data items should be equal to the number of results returned by the computational model.
	 - Syntax: `korali["Problem"]["Reference Data"][i] = myValue;`
	 - Example:
	 
```python
# Adding reference data items from a vector.

for x in refData:
 korali["Problem"]["Reference Data"] += x
```