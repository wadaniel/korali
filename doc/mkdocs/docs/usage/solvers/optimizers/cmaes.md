# Solvers / CMA-ES
   				   
##Description

This is the implementation of the CMA Evolution Strategy, as published in the [paper](https://doi.org/10.1007/3-540-32494-1_4) 

( THE REST OF THE EXPLANATION GOES HERE)


##Syntax

```python
  # Definition
  Korali["Solver"]["Method"] = "CMA-ES";
  
  # Mandatory Settings
  Korali["Solver"]["Sample Count"] = ...;
  Korali["Solver"]["Variables"][i]["Initial Mean"] = ...                # For all problem's variables
  Korali["Solver"]["Variables"][i]["Initial Standard Deviation"] = ...  # For all problem's variables

  # Optional Settings
  Korali["Solver"]["Variables"][i]["Lower Bound"] = ...
  Korali["Solver"]["Variables"][i]["Upper Bound"] = ...
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


## Plotting

Here we explain the **CMA-ES** result plot in further detail and how it can be
used to validate your optimization.

The `plot-Korali` command visualizes some of the most meaningful states of CMA-ES
stored in the json-files in the output directory (results folder).

In the figure below we see the evolution of the CMA-ES algorithm during 100
optimization steps, respectively 1000 function evaluations (here the sample size
is 10), of the negative 2-dimensional [Rosenbrock](https://en.wikipedia.org/wiki/Rosenbrock_function) function.

![figure](optimizing_rosenbrock.png)

## Description

### Quadrant 1

The first quadrant (upper left) shows 4 graphs plus markers (crosses):

* $| F |$ (red): $F$ is the best function evaluation of the current generation.
      Note that the colour of $F$ changes if $F < 0$ (red) or $F \geq 0$ (blue).
      Also, the absolute value of F is plotted since the vertical axis is given
      in log-scale.

* $\kappa(\mathbf{C})$ (cyan): This line shows the condition of the covariance matrix
	  of the proposal distribution at every generation. The condition is given
	  by the ratio of the largest Eigenvalue to the smallest Eigenvalue. A
	  large condition number may lead to numerical instabilities, this can be
	  treated by normalizing the domain of the objective variables.

* $|| \mathbf{p}_{\sigma} ||$ (black):  The evolution path is a measure of the travel
      direction of the mean of the proposal distribution of CMA-ES. The
      Euclidean norm of the evolution path plays an important rule in the Sigma
      updating rule.

* $\sigma$ (yellow) : $\sigma$ is the scaling parameter of the covariance matrix.
      The scaling parameter is updated at every generation. If Sigma becomes
	  very large or small it may have an adverse effect on the optimization.

* $| F - F_{best}|$ (crosses) : At every generation we calculate the absolute
      difference betwen the current best function evaluation ($F$) and the overall
      best found evaluation ($F_{best}$) of CMA-ES. The crosses appear only if the
      current generation does not impeove the overall result, i.e. $F < F_{best}$
      in current generation and $| F - F_{best} |$ is greater 0.

A good indicator of convergance of CMA-ES to the global maximum is given by a steady decrease of $| F - F_{best} |$.


### Quadrant 2

Objective Variables: This plot shows the evolution of the objective variables corresponding to the
evaluation of $|F|$. A line per each dimension of the optimization problem is
plotted.



### Quadrant 3

Square Root of Eigenvalues $\mathbf{C}$: The square root of the Eigenvalues of $\mathbf{C}$ are the lengths of the axes of the
(unscaled) covariance matrix. Optimally the lengths of the axes are of same magnitude.



### Quadrant 4

$\sigma \sqrt{diag(\mathbf{C})}$: the square root of the diagonal elements of the (scaled) covariance matrix of
the proposal distribution approximate the standard deviation of the parameters.
Ideally the standard deviations of all coordinates are of same magnitude.



## Example

### Shekel function

The following figure shows the results of an unsuccessful maximization of the
negative of the [Shekel](https://en.wikipedia.org/wiki/Shekel_function) function in 4
dimensions and with 10 local maxima.

[We know](http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page2354.htm)
that the Shekel function has a global minimum at (4, 4, 4, 4),
respectivel maximum in the negative case.
In quadrant 2 we see that CMA-ES converged to a different result.

In general the global optimum is not known, following observations indicate
ill convergence. Restarting CMA-ES from different starting points as well as
tuning CMA-ES internal parameters might improve optimization:

* Increasing condition (quadrant 1) of the covariance matrix of the proposal
  distribution, respectively diverging axes lenghts and standard deviations
  (quadrants 3 & 4).

* None decreasing values for $| F - F_{best} |$. Arguably CMA-ES found a better
  function evaluation on a different hill but the algorithm is trapped (the
  objective variables stabilized sampling does not overcome the saddle points).

![figure](optimizing_shekel.png)
