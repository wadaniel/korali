

In this tutorial we show how to **optimize** and **sample** the posterior
distribution of a Bayesian inference problem.







## Problem setup
In this example we will solve the inverse problem of estimating the Variables
of a linear model using noisy data. We consider the computational model,

$$
f(x\vartheta) = \vartheta_0 + \vartheta_1 x \,,
$$

for $x\in\mathbb{R}$. We assume the following error model,

$$
y = f(x\vartheta) + \varepsilon \,,
$$

with $\varepsilon$ a random variable that follows normal distribution with zero
mean and $\sigma$ standard deviation. This assumption leads to the likelihood
function,

$$
p(y|\varphix) = \mathcal{N} ( \,y \,| \, f(x\vartheta), \sigma^2 \,) \,.
$$

where $\varphi=(\vartheta,\sigma)$ is the parameter vector that contains the
computational Variables and the Variables of the statistical model.

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
p(y|\varthetax) = \prod_{i=1}^6 \mathcal{N} ( \,y_i \,| \, f(x_i\vartheta), \sigma^2 \,) \,.
$$

In order to identify the distribution of $\varphi$ conditioned on the observations $d$
we use Bayes' theorem

$$
p(\varphi | yx) = \frac{ p(y|\varphix) \, p(\varphi) }{ p(y) } \, .
$$


As a prior information we choose the uniform distribution in $[-10,10]$ for $\vartheta$
and the uniform distribution in $[0,10]$ for $\sigma$.











##  The computational model

Create a folder named `model`. Inside, create a file with name `posteriorModel.py` and paste the following code,
```python
#!/usr/bin/env python

def evaluateModel( s, x ):
 for i in range(len(x)):
  th0 = s.getVariable(0)
  th1 = s.getVariable(1)
  r  =  th0*x[i] + th1
  s.addResult(r)
```
This function corresponds implements the computational model that corresponds to $f(x\vartheta) = \vartheta_0 + \vartheta_1 x$. The object `s` must be of type `Korali::modelData`. This class provides the methods `getParameter` and `addResult`. For a detailed presentation see [here]

In the same file add the following function that return the data presented in the table above,
```python
def getReferenceData():
 x=[] y=[]
 x.append(1.0) y.append(3.2069)
 x.append(2.0) y.append(4.1454)
 x.append(3.0) y.append(4.9393)
 x.append(4.0) y.append(6.0588)
 x.append(5.0) y.append(6.8425)
 return x, y
```










## Optimization with CMA-ES


The code for this tutorial in **Python** can be found [here](https://github.com/cselab/skorali/blob/master/tutorials/a3-bayesian-inference-cmaes).



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



### The Korali object

Next we construct a `Korali` object and set the computational model,
```python
k = korali.Engine()
k.setModel(evaluateModel)
```


We add the data
```python
x, y = getReferenceData()
```

The `x` list corresponds to the *input* variables of the model. The function that
is passed to Korali should not have an argument for `x`. We have to create an intermediate
lambda function `Fx` that will hide `x` from korali.

```python
Fx = lambda koraliData: evaluateModel(koraliData, x)
```


Then, we set the model to Korali,

```python
k.setModel( Fx )
```



### The Problem type

The `Type` of the `Problem` is characterized as `Bayesian`
```python
k["Problem"] = "Bayesian"
```

When the Type is `Bayesian` we must set the type of likelihood and provide a vector with the `Reference Data` to Korali,

```python
k["Bayesian"]["Likelihood"]["Type"] = "Reference"
k["Bayesian"]["Likelihood"]["Reference Data"] = y
```

### The Variables

We define two `Variables` of type `Computational` that correspond to $\vartheta_0$ and $\vartheta_1$. The prior distribution of both is set to `Uniform`.

```python
k["Variables"][0]["Name"] = "a"
k["Variables"][0]["Bayesian"]["Type"] = "Computational"
k["Variables"][0]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][0]["Bayesian"]["Prior Distribution"]["Minimum"] = -5.0
k["Variables"][0]["Bayesian"]["Prior Distribution"]["Maximum"] = +5.0

k["Variables"][1]["Name"] = "b"
k["Variables"][1]["Bayesian"]["Type"] = "Computational"
k["Variables"][1]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][1]["Bayesian"]["Prior Distribution"]["Minimum"] = -5.0
k["Variables"][1]["Bayesian"]["Prior Distribution"]["Maximum"] = +5.0
```

The last parameter we add is of `Type` `Statistical` and corresponds to the variable
$\sigma$ in the likelihood function,

```python
k["Variables"][2]["Name"] = "Sigma"
k["Variables"][2]["Bayesian"]["Type"] = "Statistical"
k["Variables"][2]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][2]["Bayesian"]["Prior Distribution"]["Minimum"] =   0.0
k["Variables"][2]["Bayesian"]["Prior Distribution"]["Maximum"] = +5.0
```

### The Solver

Next, we choose the solver `CMA-ES`, the population size to be `12`.

```python
k["Solver"] = "CMA-ES"
k["CMA-ES"]["Objective"] = "Maximize"
k["CMA-ES"]["Sample Count"] = 12
```

And activating 3 of its available termination criteria.

```python

k["CMA-ES"]["Termination Criteria"]["Min Fitness"]["Active"] = True
k["CMA-ES"]["Termination Criteria"]["Min Fitness"]["Value"] = 1e-12

k["CMA-ES"]["Termination Criteria"]["Max Generations"]["Active"] = True
k["CMA-ES"]["Termination Criteria"]["Max Generations"]["Value"] = 100

k["CMA-ES"]["Termination Criteria"]["Max Model Evaluations"]["Active"] = True
k["CMA-ES"]["Termination Criteria"]["Max Model Evaluations"]["Value"] = 1e4
```

The CMA-ES algorithm needs the lower and upper exploration bound for each variable.

```python
k["Variables"][0]["CMA-ES"]["Lower Bound"] = -5.0
k["Variables"][0]["CMA-ES"]["Upper Bound"] = +5.0
k["Variables"][1]["CMA-ES"]["Lower Bound"] = -5.0
k["Variables"][1]["CMA-ES"]["Upper Bound"] = +5.0
k["Variables"][2]["CMA-ES"]["Lower Bound"] = 0.0
k["Variables"][2]["CMA-ES"]["Upper Bound"] = +5.0
```

For a detailed description of CMA-ES settings see [here](../../usage/solvers/optimizers/cmaes.md).

Finally, we need to add a call to the run() routine to start the Korali engine.

```python
k.run()
```

###  Run

We are now ready to run our example:

```bash
./a3-bayesian-cmaes
```

Or, alternatively:

```bash
python3 ./a3-bayesian-cmaes
```

The results are saved in the folder `_korali_result/`.



### Plot

You can see the results of CMA-ES by running the command,
```sh
python3 -m korali.plotter
```

![figure](posterior-cma.png)













## Sampling with TMCMC


The code for this tutorial in **Python** can be found [here](https://github.com/cselab/skorali/blob/master/tutorials/a3-bayesian-inference-tmcmc).


### The Solver

To sample the posterior distribution, we set the solver to `TMCMC` sampler and set a few settings,

```python
k["Solver"] = "TMCMC"
k["TMCMC"]["Covariance Scaling"] = 0.02
k["TMCMC"]["Population Size"] = 5000
k["TMCMC"]["Min Rho Update"] = 0.0
k["TMCMC"]["Coefficient of Variation"] = 0.5
k["TMCMC"]["Burn In"] = 5
```

For a detailed description of the TMCMC settings see [here](../../usage/solvers/samplers/tmcmc.md)

Finally, we need to add a call to the run() routine to start the Korali engine.

```python
k.run()
```

###  Run

We are now ready to run our example:

```bash
./a3-bayesian-tmcmc
```

Or, alternatively:

```bash
python3 ./a3-bayesian-tmcmc
```

The results are saved in the folder `_korali_result/`.



### Plot

You can see a histogram of the results by running the command
```sh
python3 -m korali.plotter
```


![figure](posterior-tmcmc.png)
