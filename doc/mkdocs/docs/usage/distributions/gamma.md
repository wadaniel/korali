# Korali / Variables / Gamma


!!! warning
    This page serves as a placeholder and is under construction.

    				   				   
##Description

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

##Syntax
       Korali["Problem"]["Variables"][i]["Distribution"] = "Gamma"

##Requirements

*{none}*

##Settings

+ Name: **Type**
     - Description: Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
     - Type: String.
	 - Allowed values:
	       - "Computational" - Determines this is a computational variable.
		   - "Statistical" - Determines this is a statistical variable.
	 - Default: "Computational"
	 - Syntax: `Korali["Problem"]["Variables"][i]["Type"] = myType`
	 - Example:

```python
# Modifying two variables to have one statistical and another computational.
Korali["Problem"]["Variables"][0]["Type"] = "Computational"
Korali["Problem"]["Variables"][1]["Type"] = "Statistical"
```

<br>

+ Name: **Minimum**
     - Description: Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
     - Type: Number.
	 - Default: *-infinity*
	 - Syntax: `Korali["Problem"]["Variables"][i]["Minimum"] = myLowerBound`
	 - Example:

```python
# Modifying the lower bound of two variables
Korali["Problem"]["Variables"][0]["Minimum"] = 0.0;
Korali["Problem"]["Variables"][1]["Minimum"] = -32.0;
```

<br>

+ Name: **Maximum**
     - Description: Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
     - Type: Number.
	 - Default: *+infinity*
	 - Syntax: `Korali["Problem"]["Variables"][i]["Maximum"] = myUpperBound`
	 - Example:

```python
# Modifying the upper bound of two variables
Korali["Problem"]["Variables"][0]["Maximum"] = 32.0;
Korali["Problem"]["Variables"][1]["Maximum"] = 0.0;
```

<br>

+ Name: **Initial Mean**
     - Description: Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
     - Type: Number.
	 - Default: *0.0*
	 - Syntax: `Korali["Problem"]["Variables"][i]["Initial Mean"] = myInitialMean`
	 - Example:

```python
# Modifying the initial guess mean of my variable
Korali["Problem"]["Variables"][0]["Initial Mean"] = 16.0;
```

<br>

+ Name: **Initial StdDev**
     - Description: Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
     - Type: Number.
	 - Default: *1.0*
	 - Syntax: `Korali["Problem"]["Variables"][i]["Initial StdDev"] = myInitialStdDev`
	 - Example:

```python
# Modifying the initial standard deviation of my variable
Korali["Problem"]["Variables"][0]["Initial StdDev"] = 2.0;
```	 

<br>

+ Name: **Min Std Change**
     - Description: Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
     - Type: Number.
	 - Default: *0.0*
	 - Syntax: `Korali["Problem"]["Variables"][i]["Min Std Change"] = myMinStdChange`
	 - Example:

```python
# Modifying the minimum delta for the standard deviation of my variable
Korali["Problem"]["Variables"][0]["Initial StdDev"] = 0.05;
```	 
