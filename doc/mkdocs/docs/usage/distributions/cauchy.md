# Distribution / Cauchy
   				   
##Description

The Cauchy distribution has the probability density function (PDF):

$$f(x;x_{0},\gamma )={\frac  {1}{\pi \gamma \left[1+\left({\frac  {x-x_{0}}{\gamma }}\right)^{2}\right]}}={1 \over \pi \gamma }\left[{\gamma ^{2} \over (x-x_{0})^{2}+\gamma ^{2}}\right],$$

where $x_{0}$ is the location parameter, specifying the **location** of the peak of the distribution, and $\gamma$ is the **scale** parameter which specifies the half-width at half-maximum (HWHM).

[Picture here?]

##Syntax

```python
  # Definition
  Korali["Problem"]["Variables"][i]["Distribution"] = "Cauchy"
  
  # Mandatory Settings
  Korali["Problem"]["Variables"][i]["Type"] = ...
  Korali["Problem"]["Variables"][i]["Location"] = ...
  Korali["Problem"]["Variables"][i]["Scale"] = ...
  
  # Optional Settings
  Korali["Problem"]["Variables"][i]["Minimum"] = ...
  Korali["Problem"]["Variables"][i]["Maximum"] = ...
  Korali["Problem"]["Variables"][i]["Initial StdDev"] = ...
  Korali["Problem"]["Variables"][i]["Min StdDev Change"] = ...
```

##Mandatory Settings

### **Type**

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

 - Value Type: *String*.
 - Allowed values:
	   - ```"Computational"``` - Determines this is a computational variable.
	   - ```"Statistical"``` - Determines this is a statistical variable.
 - Example:

```python
# Modifying two variables to have one statistical and another computational.
Korali["Problem"]["Variables"][0]["Type"] = "Computational"
Korali["Problem"]["Variables"][1]["Type"] = "Statistical"
```

### **Location**
Indicates the location of the peak of the Cauchy distribution.

- Value Type: *Real Number*
- Example: 

```python
 Korali["Problem"]["Variables"][i]["Location"] = +8.0
```

### **Scale**
Specifies the half-width at half-maximum (HWHM) of the Cauchy distribution.

 - Value Type: *Real Number*
 - Example:

```python
Korali["Problem"]["Variables"][i]["Scale"] = +2.0
```

##Optional Settings

### **Minimum**

Specifies the lower bound for the variable's value. Korali will not generate samples in which this variable falls below the specified minimum.
 
 - Value Type: *Real Number*
 - Default Value: ```-infinity```
 - Example:

```python
# Modifying the lower bound of two variables
Korali["Problem"]["Variables"][0]["Minimum"] = 0.0;
Korali["Problem"]["Variables"][1]["Minimum"] = -32.0;
```

### **Maximum**
Specifies the upper bound for the variable's value. Korali will not generate samples in which this variable falls below the specified minimum.
 
 - Value Type: *Real Number*
 - Default Value: ```+infinity```
 - Example:

```python
# Modifying the upper bound of two variables
Korali["Problem"]["Variables"][0]["Maximum"] = 32.0;
Korali["Problem"]["Variables"][1]["Maximum"] = 0.0;
```

### **Initial Mean**
Defines the initial guess mean for the variable's value. Korali will start exploring the variable space at the start of execution.

 - Value Type: *Real Number*, the value should be between the Mininum and Maximum settings.
 - Default Value: ```(Maximum+Minimum)/2```
 - Example:

```python
# Modifying the initial guess mean of my variable
Korali["Problem"]["Variables"][i]["Initial Mean"] = 16.0;
```

### **Initial StdDev**
Defines the initial guess standard deviation mean for the variable's value. Korali will generate samples around this deviation from the initial mean.

 - Value Type: *Real Number*
 - Default: ```(Maximum-Minimum)/8```
 - Example:

```python
# Modifying the initial standard deviation of my variable
Korali["Problem"]["Variables"][i]["Initial StdDev"] = 2.0;
```
	 
### **Min StdDev Change**
Defines the minimum rate of change for the standard deviation. A bigger rate may accelerate convergence, but a smaller rate will increase result precision.

 - Value Type: *Real Number*
 - Default: ```0.0```
 - Example:

```python
# Modifying the minimum delta for the standard deviation of my variable
Korali["Problem"]["Variables"][i]["Min StdDev Change""] = 0.05;
```	 
