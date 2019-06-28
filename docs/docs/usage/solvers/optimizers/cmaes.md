# Solvers / CMA-ES
   				   
##Description

### Base CMA-ES

This is the implementation of the *Covariance Matrix Adaptation Evolution Strategy*, as published in [Hansen2006](https://doi.org/10.1007/3-540-32494-1_4).

In an evolution strategy, new candidate solutions are sampled according to a multivariate normal distribution in $\mathbb {R} ^{n}$. Recombination amounts to selecting a new mean value for the distribution. Mutation amounts to adding a random vector, a perturbation with zero mean. Pairwise dependencies between the variables in the distribution are represented by a covariance matrix. The covariance matrix adaptation (CMA) is a method to update the covariance matrix of this distribution.

CMA-ES works iteratively, evaluating a number $\lambda$ of samples per generation, and improving the covariance matrix for the samples in the next generation.

**Base Requirements:**

+ The *Sample Count* $\lambda$ needs to be defined.
+ The *Initial Mean* needs to be defined for every variable.
+ The *Initial Standard Deviation* needs to be defined for every variable.

### Constrained CMA-ES

This solver also implements the *Constrained Covariance Matrix Adaptation Evolution Strategy*, as published in [Arampatzis2019](https://dl.acm.org/citation.cfm?doid=3324989.3325725).

CCMA-ES is an extension of [CMA-ES](/usage/solvers/optimizers/cmaes/) for constrained optimization problems. It uses the principle of *viability boundaries* to find an initial mean vector for the proposal distribution that does not violate constraints, and secondly it uses a  *constraint handling technique* to efficiently adapt the proposal distribution to the constraints.

** Constraint Requirements:**

+ At least one *Constraint function* defined.
+ The *Viability Sample Count* $via-\lambda$ needs to be defined.
+ The *Initial Mean* needs to be defined for every variable.
+ The *Initial Standard Deviation* needs to be defined for every variable.strained

##Solver Settings

??? abstract "Result Output Frequency" 

	Specifies the output frequency of intermediate result files. By default, Korali sets this value to *1* (i.e. each generation). Example:

	```python
    # Reduce the number of result files
	k["CMA-ES"]["Result Output Frequency"] = 10
	```

??? abstract "Objective" 

	Specifies whether the problem evaluation is to be *minimized* or *maximized*. By default, Korali sets this value to *Maximize*. Example:

	```python
	#Maximizing problem evaluation (default)
	k["CMA-ES"]["Objective"] = "Maximize"

	#Minimizing problem evaluation
	k["CMA-ES"]["Objective"] = "Minimize"
	```

??? abstract "Sample Count" 

	Specifies the number of samples $\lambda$ to evaluate per generation. Example:

	```python
    # Specifying Sample Count
	k["CMA-ES"]["Sample Count"] = 32
	```

??? abstract "Sigma Cumulation Factor" 

	Controls the learning rate of the conjugate evolution path for $\sigma$ update. By default, Korali sets this value to $\frac{\mu_{eff}+2}{\mu_{eff}+N+3}$. Example:

	```python
    # Increasing Sigma Cumulation Factor
	k["CMA-ES"]["Sigma Cumulation Factor"] = 1.0
	```

??? abstract "Damp Factor" 

	Controls the update of the scaling factor $\sigma$. By default, Korali calibrates this value from $\mu_{eff}$ and number of variables (must be larger 0.0). Example:

	```python
    # Increasing Damp Factor
	k["CMA-ES"]["Damp Factor"] = 1.0
	```	
	
??? abstract "Max Resamplings" 

	Number of resamplings per candidate per generation if sample is outside of Lower and Uppder Bound. By default, Korali sets this value to $10^6$. Example:

	```python
    # Increasing Max Resamplings
	k["CMA-ES"]["Max Resamplings"] = 10e7
	```	

??? abstract "Sigma Bounded" 

	Sets an upper bound for scaling factor $\sigma$. The upper bound is given by the average of the initial standard deviation of the variables. By default, Korali setis this flag to *False*. Example:

	```python
    # Set an upper limit for Sigma
	k["CMA-ES"]["Sigma Bounded"] = True
	```	
	
??? abstract "Mu Value" 

	Number of best samples used to update the covariance matrix and the mean. By default, Korali sets this value to $\lambda / 2$. Example:

	```python
    # Specifying Mu
	k["CMA-ES"]["Mu"]["Value"] = 8
	```	

??? abstract "Mu Type" 

	Weights given to the Mu best values to update the covariance matrix and the mean. By default, Korali sets this value to *Logarithmic*. Example:

	```python
	k["CMA-ES"]["Mu"]["Type"] = "Equal"
	k["CMA-ES"]["Mu"]["Type"] = "Linear"
	k["CMA-ES"]["Mu"]["Type"] = "Logarithmic"
	```

??? abstract "Cumulative Covariance" 

	Controls the learning rate of the evolution path for the covariance update. By default, Korali calibrates this value from $\mu_{eff}$ and number of variables (must be in $(0,1]$). Example:

	```python
    # Increasing Cumulative Covariance
	k["CMA-ES"]["Covariance Matrix"]["Cumulative Covariance"] = 0.9
	```

??? abstract "Is Diag" 

	Tells the solver to treat the covariance matrix as diagonal in order to speed up covariance matrix updates. By default, Korali sets this flag to *False*. Example:

	```python
    # Diagonal Covariance Matrix
	k["CMA-ES"]["Covariance Matrix"]["Is Diag"] = True
	```

## Constraint Settings

??? abstract "Viability Sample Count" 

	Specifies the number of samples ($via-\lambda$) used per generation during the viability regime, i.e. during the search for a mean vector $\bf{m}$ not violating the constraints. Recomended is a small population size. Example:

	```python
    # Setting Viability Sample Count
	k["CCMA-ES"]["Viability Sample Count"] = 2

	```

??? abstract "Mu Viability" 

	Number of best samples used to update the covariance matrix and the mean during the viability regime. By default, Korali sets this value to $via-\lambda / 2$. Example:

	```python
    # Setting Mu Viability
	k["CCMA-ES"]["Mu"]["Viability"] = 1

	```

??? abstract "Target Success Rate" 

	Controls the updates of the scaling factor $\sigma$ during the viability regime. By default, Korali sets this value to $2/11$. Example:

	```python
    # Increasing Target Success Rate
	k["CCMA-ES"]["Target Success Rate"] = 0.6

	```

??? abstract "Adaption Size" 

	Controls the covariane matrix adaption strength if samples violate constraints. By default, Korali sets this value to $0.1$. Example:

	```python
    # Increasing Adaption Size
	k["CCMA-ES"]["Adaption Size"] = 0.3

	```

??? abstract "Max Corrections" 

	Max number of covairance matrix adaptions per generation during the constraint handling loop. By default, Korali sets this value to $10^6$. Example:

	```python
    # Increasing Max Corrections
	k["CCMA-ES"]["Max Corrections"] = 10e9

	```

??? abstract "Normal Vector Learning Rate" 

	Learning rate of constraint normal vectors (must be in $(0,1])$. By default, Korali sets this value to $\frac{1}{\lambda+2}$. Example:

	```python
    # Increasing Normal Vector Learning Rate
	k["CCMA-ES"]["Normal Vector Learning Rate"] = 0.7

	```

??? abstract "Global Success Learning Rate" 

	Learning rate of success factor of objective function improvements. Required for scaling factor $\sigma$ update. By default, Korali sets this value to $\frac{1}{12}$. Example:

	```python
    # Increasing Global Success Learning Rate
	k["CCMA-ES"]["Global Success Learning Rate"] = 0.5
	```
		 
## Termination Criteria

??? abstract "Max Generations" 

	Specifies the maximum number of generations to run. By default, Korali sets this criterion as active and its value to *1000*. Example:

	```python
    # Increase number of Generations to run
	k["CMA-ES"]["Termination Criteria"]["Max Generations"]["Active"] = True
	k["CMA-ES"]["Termination Criteria"]["Max Generations"]["Value"]  = 5000
	```
	
??? abstract "Max Model Evaluations" 

	Specifies the maximum number of objective function evaluations. By default, Korali sets this criterion as inactive. Example:

	```python
    # Specifying Max Model Evaluations
	k["CMA-ES"]["Termination Criteria"]["Max Model Evaluations"]["Active"] = True
	k["CMA-ES"]["Termination Criteria"]["Max Model Evaluations"]["Value"]  = 50000
	```	
	
??? abstract "Fitness" 

	Specifies the goal fitness to stop execution. By default, Korali sets this criterion as inactive. Example:

	```python
    # Specifying Min Fitness
	k["CMA-ES"]["Termination Criteria"]["Min Fitness"]["Active"] = True
	k["CMA-ES"]["Termination Criteria"]["Min Fitness"]["Value"]  = 0.0
	```	
	
??? abstract "Fitness Diff Threshold" 

	Specifies the minimum fitness differential between two consecutive generations before stopping execution. By default, Korali sets this criterion as active and value $1^-9$. Example:

	```python
    # Decreasing Fitness Diff Threshold
	k["CMA-ES"]["Termination Criteria"]["Fitness Diff Threshold"]["Active"] = True
	k["CMA-ES"]["Termination Criteria"]["Fitness Diff Threshold"]["Value"]  = 1e-12
	```	

??? abstract "Min Standard Deviation" 

	Specifies a minimum standard deviation for the proposal distribution of the variables. Korali terminates if this value is undershot by all variables. By default, Korali sets this criterion as active and value $1^-12$. Example:

	```python
    # Increasing Min Standard Deviation
	k["CMA-ES"]["Termination Criteria"]["Min Standard Deviation"]["Active"] = True
	k["CMA-ES"]["Termination Criteria"]["Min Standard Deviation"]["Value"]  = 1e-9
	```	
	
??? abstract "Max Standard Deviation" 

	Specifies the maximum allowed increase in the standard deviation for the proposal distribution of a variable. By default, Korali sets this criterion as active and value $1^18$. Example:

	```python
    # Decreasing Max Standard Deviation
	k["CMA-ES"]["Termination Criteria"]["Max Standard Deviation"]["Active"] = True
	k["CMA-ES"]["Termination Criteria"]["Max Standard Deviation"]["Value"]  = 1e12
	```	

??? abstract "Max Condition Covariance" 

	Specifies the maximum condition of the covariance matrix. By default, Korali sets this criterion as active and value $1^18$. Example:

	```python
    # Decreasing Max Standard Deviation
	k["CMA-ES"]["Termination Criteria"]["Max Condition Covariance"]["Active"] = True
	k["CMA-ES"]["Termination Criteria"]["Max Condition Covariance"]["Value"]  = 1e15
	```	
	
## Variable Settings


??? abstract "Minimum" 

	Specifies the lower bound for the variable's value. Korali will not generate samples for which this variable falls below the specified minimum. By default, Korali sets this value to *-Infinity*. Example:

	```python
	# Modifying the lower bound of two variables
	k["Variables"][0]["CMA-ES"]["Minimum"] = -32.0
	k["Variables"][1]["CMA-ES"]["Minimum"] = -32.0;
	```

??? abstract "Maximum" 

	Specifies the upper bound for the variable's value. Korali will not generate samples for which this variable falls below the specified maximum. By default, Korali sets this value to *+Infinity*. Example:

	```python
	# Modifying the upper bound of two variables
	k["Variables"][0]["CMA-ES"]["Maximum"] = 32.0;
	k["Variables"][1]["CMA-ES"]["Maximum"] = 32.0;
	```

??? abstract "Initial Mean" 

	Defines the initial mean for the proposal distribution. This value must be defined between the variable's *Mininum* and *Maximum* settings By default, Korali sets this value in the center of the domain. Example:

	```python
	# Modifying the Initial Mean ean of variable 0 and 1
	k["Variables"][0]["CMA-ES"]["Initial Mean"] = 16.0;
	k["Variables"][1]["CMA-ES"]["Initial Mean"] = 16.0;
	```

??? abstract "Initial Standard Deviation" 

	Defines the initial standard deviation of the proposal distribution for a variable. By default, Korali sets this value to 30% of the domain width. Example:

	```python
	# Modifying the Initial Standard Deviation of variable 0 and 1
	k["Variables"][0]["CMA-ES"]["Initial Standard Deviation"] = 2.0;
	k["Variables"][1]["CMA-ES"]["Initial Standard Deviation"] = 2.0;
	```

??? abstract "Minimum Standard Deviation Changes" 

	Defines a lower bound for the standard deviation updates of the proposal distribution for a variable. Korali increases the scaling factor $\sigma$ if this value is undershot. By default, Korali sets this value to 0.0 (inactive). Example:

	```python
	# Specifying Minum Standard Deviation Changes
	k["Variables"][0]["CMA-ES"]["Minimum Standard Deviation Changes"] = 1e-6
	k["Variables"][1]["CMA-ES"]["Minimum Standard Deviation Changes"] = 1e-6
	```

??? abstract "Log Space" 

	Specifies if the log space of a variable should be searched. By default, Korali sets this flag to $False$.
		
    ```python
	# Search in Log Space
	k["Variables"][i]["CMA-ES"]["Log Space"] = True
	```

## Plotting

Here we explain the **CMA-ES** result plot in further detail and how it can be
used to validate your optimization.

The module korali.plotter (run with `python3 -m korali.plotter`) command visualizes some of the most meaningful states of CMA-ES
stored in the result files in the output directory (`_korali_result`).
To plot a running simulation use the command `python3 -m korali.plotter --live` for incremental plots.

In the figure below we see the evolution of the CMA-ES algorithm during 100
optimization steps, respectively 1000 function evaluations (here the sample size
is 10), of the negative 2-dimensional [Rosenbrock](https://en.wikipedia.org/wiki/Rosenbrock_function) function.

![figure](optimizing_rosenbrock.png)

**Quadrant 1**: The first quadrant (upper left) shows 4 graphs plus markers (crosses):

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

**Quadrant 2**: Objective Variables: This plot shows the evolution of the objective variables corresponding to the
evaluation of $|F|$. A line per each dimension of the optimization problem is plotted.


**Quadrant 3**: Square Root of Eigenvalues $\mathbf{C}$: The square root of the Eigenvalues of $\mathbf{C}$ are the lengths of the axes of the (unscaled) covariance matrix. Optimally the lengths of the axes are of same magnitude.

**Quadrant 4**: $\sigma \sqrt{diag(\mathbf{C})}$: the square root of the diagonal elements of the (scaled) covariance matrix of
the proposal distribution approximate the standard deviation of the parameters. Ideally the standard deviations of all coordinates are of same magnitude.



**Example: Shekel function**

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
