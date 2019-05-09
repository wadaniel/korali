

In this tutorial we show how to **optimize** and **sample** directly from a
given function.

The code in **Python** for this tutorial can be found [here](https://github.com/cselab/skorali/blob/master/examples/python/quick_start/direct.py) and the code in **C++** can be found [here](https://github.com/cselab/skorali/blob/master/examples/cxx/quick_start/direct.cpp).



## Problem setup
We are given the function $g(\vartheta)=\exp(-\vartheta^2)$ for $\vartheta\in[-10,10]$.

First we want to find the maximum of the function in the given interval.

!!! tip
    Since $\exp$ is monotonically increasing function, g has the same optima
    as $\log g$.

Next, we assume that $f$ represents the *unnormalized* density of a distribution.
We want to draw samples from this distribution.

!!! note
    For stability reasons, the sampling algorithm works with the $\log$ of an unnormalized density.

For the rest of the tutorial we will work with the function $f(\vartheta)=\log g(\vartheta) = -\vartheta^2$.


## Optimize

First import the needed modules
```python
#!/usr/bin/env python
import sys
import threading
import libkorali
```

###  The computational model
Then write the code for $f$
```python
def F( s ):
    x = s.getParameter(0)
    r = -x*x
    s.addResult(r)
```


###  The korali object

Next we construct a `korali` object using the function `F`
```python
korali = libkorali.Engine( F )
```

###  The Problem type
Then, we set the type of the problem to `Direct Evaluation`
```python
korali["Problem"]["Objective"] = "Direct Evaluation"
```

###  The Parameters
```python
korali["Parameters"][0]["Name"] = "X"
korali["Parameters"][0]["Distribution"] = "Uniform"
korali["Parameters"][0]["Type"] = "Computational"
korali["Parameters"][0]["Minimum"] = -10.0
korali["Parameters"][0]["Maximum"] = +10.0
```

###  The Solver
```python
korali["Solver"]["Method"] = "CMA-ES"
korali["Solver"]["Termination Criteria"]["Min DeltaX"] = 1e-8
korali["Solver"]["Termination Criteria"]["Max Generations"] = 100
korali["Solver"]["Lambda"] = 5
```

###  Run
```python
korali["Seed"] = 0xC0FFEE;
korali["Verbosity"] = "Detailed";
```


```python
korali.run();
```


###  Plot

```sh
plot_cma.py korali0
```

![figure](direct-cma.png)






## Sample

###  The Solver

```python
korali["Solver"]["Method"] = "TMCMC"
korali["Solver"]["Covariance Scaling"] = 0.02
korali["Solver"]["Population Size"] = 5000
korali["Solver"]["Burn In"] = 5
```

###  Plot

```sh
plot_tmcmc.py korali0
```

![figure](direct-tmcmc.png){: .center}


??? tip "Plot with Matlab"
    Assume that the 7th is last generation. In order to read the json file extract
    the population run these commands:
    ```
    s=jsondecode(fileread('s00007.json'));
    a=s.Solver.State.DatabasePoints;
    ```
    Then you can plot the histogram of the samples against the density of
    $\mathcal{N}(0,1/\sqrt{2})$:
    ```
    h=histogram(a,50);
    h.Normalization='pdf';
    hold on;
    x=-3:0.01:3;
    p=plot(x,normpdf(x,0,sqrt(0.5)));
    p.LineWidth = 4;
    ax=gca;
    ax.XLim=[-3,3];
    ax.FontSize=16;
    grid on;
    ```
