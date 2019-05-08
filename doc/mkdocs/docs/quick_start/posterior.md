

In this tutorial we show how to **optimize** and **sample** the posterior
distribution of a Bayesian inference problem.


The code for this tutorial in **Python** can be found [here](https://github.com/cselab/skorali/blob/master/examples/python/quick_start/posterior.py) and the code in **C++** can be found [here](https://github.com/cselab/skorali/blob/master/examples/cxx/quick_start/posterior.cpp).



## Problem setup
In this example we will solve the inverse problem of estimating the parameters
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
p(y|\varphi;x) = \mathcal{N} ( \,y \,| \, f(x;\vartheta), \sigma^2 \,) \,.
$$

where $\varphi=(\vartheta,\sigma)$ is the parameter vector that contains the
computational parameters and the parameters of the statistical model.

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
p(y|\vartheta;x) = \prod_{i=1}^6 \mathcal{N} ( \,y_i \,| \, f(x_i;\vartheta), \sigma^2 \,) \,.
$$

In order to identify the distribution of $\varphi$ conditioned on the observations $d$
we use Bayes' theorem

$$
p(\varphi | y;x) = \frac{ p(y|\varphi;x) \, p(\varphi) }{ p(y) } \, .
$$


As a prior information we choose the uniform distribution in $[-10,10]$ for $\vartheta$
and the uniform distribution in $[0,10]$ for $\sigma$.







## Optimize
First import the needed modules
```python
#!/usr/bin/env python
import sys
import threading
import libkorali
```


### The computational model

```python
def F( s, x ):
  for i in range(len(x)):
    th0 = s.getParameter(0)
    th1 = s.getParameter(1)
    r  =  th0*x[i] + th1
    s.addResult(r)

```

### The data
```python
x=[];            y=[];
x.append(1.0);   y.append(3.2069);
x.append(2.0);   y.append(4.1454);
x.append(3.0);   y.append(4.9393);
x.append(4.0);   y.append(6.0588);
x.append(5.0);   y.append(6.8425);
```

### The Korali object
```python
Fx = lambda s: F( s, x )

korali = libkorali.Engine( Fx )
```

or `Fx = lambda s,*,x=x: F( s, x )`


### The problem type
```python
korali["Problem"]["Objective"] = "Posterior";
```

```python
for i in range(len(y)):
  korali["Problem"]["Reference Data"][i] = y[i];
```

### The parameters
```python
  korali["Parameters"][0]["Name"] = "a";
  korali["Parameters"][0]["Type"] = "Computational";
  korali["Parameters"][0]["Distribution"] = "Uniform";
  korali["Parameters"][0]["Minimum"] = -10.0;
  korali["Parameters"][0]["Maximum"] = +10.0;

  korali["Parameters"][1]["Name"] = "b";
  korali["Parameters"][1]["Type"] = "Computational";
  korali["Parameters"][1]["Distribution"] = "Uniform";
  korali["Parameters"][1]["Minimum"] = -10.0;
  korali["Parameters"][1]["Maximum"] = +10.0;
```

```python
  korali["Parameters"][2]["Name"] = "Sigma";
  korali["Parameters"][2]["Type"] = "Statistical";
  korali["Parameters"][2]["Distribution"] = "Uniform";
  korali["Parameters"][2]["Minimum"] = 0.0;
  korali["Parameters"][2]["Maximum"] = 20.0;
```

### The solver
```python
  korali["Solver"]["Method"] = "CMA-ES";
  korali["Solver"]["Lambda"] = 12;
  korali["Solver"]["Termination Criteria"]["Min DeltaX"] = 1e-11;
  korali["Solver"]["Termination Criteria"]["Min Fitness"] = 1e-12;
  korali["Solver"]["Termination Criteria"]["Max Generations"] = 1e4;
  korali["Solver"]["Termination Criteria"]["Max Model Evaluations"] = 1e4;
```

### Run
```python
  korali["Seed"] = 0xC0FFEE;
  korali["Verbosity"] = "Detailed";

  korali.run();
```






##Sample
```python
  korali["Solver"]["Method"] = "TMCMC";
  korali["Solver"]["Covariance Scaling"] = 0.02;
  korali["Solver"]["Population Size"] = 5000;
  korali["Solver"]["Burn In"] = 5;
  korali["Solver"]["Coefficient of Variation"] = 0.5;
```
