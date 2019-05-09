

In this tutorial we show how to **optimize** and **sample** directly from a
given function.

The code in **Python** for this tutorial can be found [here](https://github.com/cselab/sKorali/blob/master/examples/python/quick_start/direct.py) and the code in **C++** can be found [here](https://github.com/cselab/skorali/blob/master/examples/cxx/quick_start/direct.cpp).



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
import libKorali
```

###  The computational model
Then write the code for $f$
```python
def F( s ):
    x = s.getParameter(0)
    r = -x*x
    s.addResult(r)
```

The argument of `F` must be of type `Korali::modelData`. This class provides the methods
`getParameter` and `addResult`. For a detailed presentation see [here]



###  The Korali object

Next we construct a `Korali` object using the function `F`
```python
Korali = libkorali.Engine( F )
```

###  The Problem type
Then, we set the type of the problem to `Direct Evaluation`
```python
Korali["Problem"]["Type"] = "Direct Evaluation"
```

###  The Variables
In this problem there is only one parameter that takes values in $[-10,10]$.
```python
Korali["Problem"]["Variables"][0]["Name"] = "X"
Korali["Problem"]["Variables"][0]["Distribution"] = "Uniform"
Korali["Problem"]["Variables"][0]["Minimum"] = -10.0
Korali["Problem"]["Variables"][0]["Maximum"] = +10.0
```


###  The Solver
Next, we choose the solver `CMA-ES`, the population size to be `5` and set
two termination criteria,

```python
Korali["Solver"]["Method"] = "CMA-ES"
Korali["Solver"]["Lambda"] = 5
Korali["Solver"]["Termination Criteria"]["Min DeltaX"] = 1e-8
Korali["Solver"]["Termination Criteria"]["Max Generations"] = 100
```
For a detailed description of CMA-ES settings see [here](../../usage/solvers/optimizers/cmaes.md).

###  Run
We set the `Seed` to a fixed value and the `Verbosity` level to the maximum available,

```python
Korali["Seed"] = 0xC0FFEE;
Korali["Verbosity"] = "Detailed";
```

Finally, we are ready to run the simulation,

```python
Korali.run();
```

The results are saved in the folder `_Korali_result/`.


###  Plot

You can see the results of CMA-ES by running the command,
```sh
Korali-plot
```

![figure](direct-cma.png)






## Sample

###  The Solver

We set the solver to `TMCMC` sampler and set a few settings,

```python
Korali["Solver"]["Method"] = "TMCMC"
Korali["Solver"]["Covariance Scaling"] = 0.02
Korali["Solver"]["Population Size"] = 5000
Korali["Solver"]["Burn In"] = 5
```

For a detailed description of the TMCMC settings see
[here](../../usage/solvers/samplers/tmcmc.md)


###  Plot
You can see a histogram of the results by running the command
```sh
Korali-plot
```

The next figure is plotted in Matlab and shows the histogram of the samples superimposed
with the density function of $\mathcal{N}(0,1/\sqrt{2})$.


??? tip "Plot with Matlab"
    Assume that the 7th is last generation. In order to read the json file extract
    the population run these commands:
    ```
    s = jsondecode(fileread('s00007.json'));
    a = s.Solver.State.DatabasePoints;
    ```
    Then you can plot the histogram of the samples against the density of
    $\mathcal{N}(0,1/\sqrt{2})$:
    ```
    h = histogram(a,50);
    h.Normalization='pdf';
    hold on;

    x = -3:0.01:3;
    p = plot(x,normpdf(x,0,sqrt(0.5)));
    p.LineWidth = 4;

    ax=gca;
    ax.XLim=[-3,3];
    ax.FontSize=16;
    grid on;
    ```

![figure](direct-tmcmc.png){: .center}
