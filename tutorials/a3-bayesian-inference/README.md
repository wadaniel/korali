# A.3 - Inferring Parameter Values with Bayes' Theorem

In this tutorial we show how to **optimize** and **sample** the posterior
distribution of a Bayesian inference problem. We will go through the code sample
 [here](run-cmaes.py) first, and then modify it to get [this code sample](run-tmcmc.py).


## Problem Setup
In this example we will solve the inverse problem of estimating the Variables
of a linear model using noisy data. We consider the computational model,

$$
f(x;\vartheta) = \vartheta_0 + \vartheta_1 x \,,
$$

for $x\in\mathbb{R}$. We assume the following error model,

$$
y = f(x;\vartheta) + \varepsilon \,,
$$

with $\varepsilon$ a random variable that follows normal distribution with zero
mean and $\sigma$ standard deviation. This assumption leads to the likelihood
function,

$$
p(y|\varphi,x) = \mathcal{N} ( \,y \,| \, f(x;\vartheta), \sigma^2 \,) \,.
$$

where $\varphi=(\vartheta,\sigma)$ is the parameter vector that contains the
computational variables and the variables of the statistical model.

We assume that we are given the following data set:

<center>

| $x_i$  | $y_i$  |
|:-:|:-:|
| 1  | 3.2069  |
| 2  | 4.1454  |
| 3  | 4.9393  |
| 4  | 6.0588  |
| 5  | 6.8425  |

</center>

We call this data set $d=\{x_i,y_i\}_{i=1}^5$. Assuming that each datum is
independent, the likelihood of $d$ under the linear model is given by

$$
p(y|\vartheta,x) = \prod_{i=1}^6 \mathcal{N} ( \,y_i \,| \, f(x_i,\vartheta), \sigma^2 \,) \,.
$$

In order to identify the distribution of $\varphi$ conditioned on the observations $d$
we use Bayes' theorem

$$
p(\varphi | y,x) = \frac{ p(y|\varphi,x) \, p(\varphi) }{ p(y) } \, .
$$


As a prior information we choose the uniform distribution in $[-5,5]$ for $\vartheta$
and the uniform distribution in $[0,10]$ for $\sigma$.


##  The Objective Function

Create a folder named `model`. Inside, create a file with name `posteriorModel.py` and paste the following code,
```python
#!/usr/bin/env python

def model( s, x ):

    v1 = s["Parameters"][0]
    v2 = s["Parameters"][1]

    result = [ ]
    for i in range(len(x)):
      result.append(v1*x[i] + v2)

    s["Reference Evaluations"] = result
```
This function corresponds implements the computational model that corresponds to $f(x\vartheta) = \vartheta_0 + \vartheta_1 x$. Note: The following might be outdated: The object `s` must be of type `Korali::modelData` This class provides the methods `getParameter` and `addResult`. For a detailed presentation see [here]

In the same file add the following functions that return the data presented in the table above,
```python
def getReferenceData():
 y=[]
 y.append(3.2069);
 y.append(4.1454);
 y.append(4.9393);
 y.append(6.0588);
 y.append(6.8425);
 return y

def getReferencePoints():
 x=[]
 x.append(1.0);
 x.append(2.0);
 x.append(3.0);
 x.append(4.0);
 x.append(5.0);
 return x
```

## Optimization with CMA-ES


First, open a file and import the korali module
```python
#!/usr/bin/env python3
import korali
```
Import the computational model,
```python
import sys
sys.path.append('./model')
from posteriorModel import *
```

### The Korali Experiment Object

Next we construct a `Korali.Experiment` object and set the computational model, where we already pass the data,

```python
e = korali.Experiment()
e["Problem"]["Computational Model"] = lambda sampleData: model(sampleData, getReferencePoints())

```

The reference points `x` returned by `getReferencePoints()` correspond to the *input* variables of the model. The function that
is passed to Korali should not have an argument for `x`. We have to create an intermediate
lambda function that will hide `x` from korali.

```python
lambda sampleData: model(sampleData, getReferencePoints())
```


### The Problem Type

The `Type` of the `Problem` is characterized as `Bayesian`
```python
e["Problem"]["Type"] = "Evaluation/Bayesian/Inference/Reference"
```

When the Type is `Bayesian` we must set the type of likelihood and provide a vector with the `Reference Data` to Korali,

```python
e["Problem"]["Likelihood Model"] = "Additive Normal"
e["Problem"]["Reference Data"] = getReferenceData()
```

### The Variables

We define two `Variables` of type `Computational` that correspond to $\vartheta_0$ and $\vartheta_1$. The prior distribution of both is set to `Uniform`.

```python
e["Variables"][0]["Name"] = "a"
e["Variables"][0]["Bayesian Type"] = "Computational"
e["Variables"][0]["Prior Distribution"] = "Uniform 0"
e["Variables"][0]["Initial Mean"] = +0.0
e["Variables"][0]["Initial Standard Deviation"] = +1.0

e["Variables"][1]["Name"] = "b"
e["Variables"][1]["Bayesian Type"] = "Computational"
e["Variables"][1]["Prior Distribution"] = "Uniform 1"
e["Variables"][1]["Initial Mean"] = +0.0
e["Variables"][1]["Initial Standard Deviation"] = +1.0

```

The last parameter we add is of `Type` `Statistical` and corresponds to the variable
$\sigma$ in the likelihood function,

```python
e["Variables"][2]["Name"] = "Sigma"
e["Variables"][2]["Bayesian Type"] = "Statistical"
e["Variables"][2]["Prior Distribution"] = "Uniform 2"
e["Variables"][2]["Initial Mean"] = +2.5
e["Variables"][2]["Initial Standard Deviation"] = +0.5
```



```python

```

### The Solver

Next, we choose the solver `CMA-ES`, the population size to be `24`.

```python

e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = 24
```

And activating one of its available termination criteria.

```python

e["Solver"]["Termination Criteria"]["Max Generations"] = 100

```

We also need to configure the problem's random distributions, which we referred to when defining our variables,

```python
e["Distributions"][0]["Name"] = "Uniform 0"
e["Distributions"][0]["Type"] = "Univariate/Uniform"
e["Distributions"][0]["Minimum"] = -5.0
e["Distributions"][0]["Maximum"] = +5.0

e["Distributions"][1]["Name"] = "Uniform 1"
e["Distributions"][1]["Type"] = "Univariate/Uniform"
e["Distributions"][1]["Minimum"] = -5.0
e["Distributions"][1]["Maximum"] = +5.0

e["Distributions"][2]["Name"] = "Uniform 2"
e["Distributions"][2]["Type"] = "Univariate/Uniform"
e["Distributions"][2]["Minimum"] = 0.0
e["Distributions"][2]["Maximum"] = +5.0
```

For a detailed description of CMA-ES settings see [here](../../usage/solvers/cmaes.md).

Finally, we configure the output, and then need to add a call to the run() routine to start the Korali engine.

```python
e["Results"]["Frequency"] = 5
e["Console"]["Frequency"] = 5

k = korali.Engine()
k.run(e)
```

###  Running

We are now ready to run our example:

```bash
./a3-bayesian-cmaes
```

Or, alternatively:

```bash
python3 ./a3-bayesian-cmaes
```

The results are saved in the folder `_korali_result/`.



### Plotting

You can see the results of CMA-ES by running the command,
```sh
python3 -m korali.plotter
```

![figure](posterior-cma.png)


## Sampling with TMCMC


### The Solver

To sample the posterior distribution, we set the solver to `TMCMC` sampler and set a few settings,

```python
e["Solver"]["Type"] = "Sampler/TMCMC"
e["Solver"]["Population Size"] = 5000
```

For a detailed description of the TMCMC settings see [here](../../usage/solvers/tmcmc.md)

Finally, we need to add a call to the run() routine to start the Korali engine.

```python
k.run(e)
```

###  Running

We are now ready to run our example:

```bash
./a3-bayesian-tmcmc
```

Or, alternatively:

```bash
python3 ./a3-bayesian-tmcmc
```

The results are saved in the folder `_korali_result/`.

### Plottting

You can see a histogram of the results by running the command
```sh
python3 -m korali.plotter
```

![figure](posterior-tmcmc.png)
