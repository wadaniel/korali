# Introduction

!!! warning
    This page serves as a placeholder and is under construction.


##Korali

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

##Modules

![](modules.png)

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

###Problem

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

###Variables

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

###Model

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.


###Conduit

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

##Syntax

###Python

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

###C++

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

##Requirements

+ It should contain at least one computational variable as input for the computational model.
+ It should contain at least one statistical variable for the value of sigma in the Gaussian error calculation.

##Settings

+ Name: **Seed**
     - Description: Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
     - Type: Number
	 - Default: *0*
	 - Syntax: `Korali["Seed"] = seed`
	 - Example:

```python
# Defining a specific random seed
Korali["Seed"] = 0xC0FFEE
```

<br>

+ Name: **Vebosity**
     - Description: Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
     - Type: String.
	 - Allowed values:
	       - "Silent" - Prints no results.
		   - "Minimal" - Prints minimal progress information.
		   - "Normal" - Standard verbosity levels.
		   - "Detailed" - Prints details about the results and execution state.
	 - Default: "Computational"
	 - Syntax: `Korali["Verbosity"] = verbosityLevel`
	 - Example:

```python
# Defining a detailed verbosity level.
Korali["Verbosity"] = "Detailed"
```
