

In this tutorial we show how to **sample** directly from a given function.


## Problem setup
We are given the function $g(\vartheta)=\exp(-\vartheta^2)$ for $\vartheta\in[-10,10]$.

We assume that $f$ represents the *unnormalized* density of a distribution.
We want to draw samples from this distribution.

!!! note
    For stability reasons, the sampling algorithm works with the $\log$ of an unnormalized density.

For the rest of the tutorial we will work with the function $f(\vartheta)=\log g(\vartheta) = -\vartheta^2$.





##  The computational model
Create a folder named `model`. Inside, create a file with name `directModel.py` and paste the following code,
```python
#!/usr/bin/env python

def evaluateModel( s ):
  x = s.getVariable(0)
  r = -x*x
  s.addResult(r)
```
This is the computational model.




## Sampling with MCMC

The code in **Python** for this tutorial can be found [here](https://github.com/cselab/skorali/blob/master/tutorials/a2-sampling-mcmc).

First, open a file and import the korali module
```python
#!/usr/bin/env python3
import korali
```
Import the computational model,
```python
import sys
sys.path.append('./model')
from directModel import *
```


###  The Korali object

Next we construct a `Korali` object and set the computational model,
```python
k = korali.Engine()
k.setModel(evaluateModel)
```

###  The Problem type
Then, we set the type of the problem to `Direct Evaluation`
```python
k["Problem"] = "Direct Evaluation"
```

###  The Variables
In this problem there is only one variable,
```python
k["Variables"][0]["Name"] = "X";
```


###  The Solver
We choose the solver `MCMC` and set the initial mean and standard deviation  of the parameter `X`.
```python
k["Solver"] = "MCMC";

k["Variables"][0]["MCMC"]["Initial Mean"] = 0.0;
k["Variables"][0]["MCMC"]["Initial Standard Deviation"] = 1.000;

k["MCMC"]["Population Size"] = 5000;
k["MCMC"]["Burn In"] = 500;
k["MCMC"]["Adaptive Sampling"]  = True;
k["MCMC"]["Result Output Frequency"]  = 5000;
```
We also set some settings for MCMC. For a detailed description of the MCMC settings see
[here](../../usage/solvers/samplers/mcmc.md).



###  Run

Finally, we are ready to run the simulation,

```python
k.run()
```

The results are saved in the folder `_korali_result/`.


###  Plot
You can see a histogram of the results by running the command
```sh
python3 -m korali.plotter
```











## Sampling with TMCMC
The code in **Python** for this tutorial can be found [here](https://github.com/cselab/skorali/blob/master/tutorials/a2-sampling-tmcmc).

First, open a file and import the korali module
```python
#!/usr/bin/env python3
import korali
```
Import the computational model,
```python
import sys
sys.path.append('./model')
from directModel import *
```


###  The Korali object

Next we construct a `Korali` object and set the computational model,
```python
k = korali.Engine()
k.setModel(evaluateModel)
```

###  The Problem type
Then, we set the type of the problem to `Direct Evaluation`
```python
k["Problem"] = "Bayesian"
```
Notice that although the problem is "Direct" we define it as "Bayesian". The reason is
that TMCMC is designed for Bayesian problems. But we can always transform a "Direct" problem into "Bayesian" by using a uniform prior distribution and a likelihood the function we want to sample.

Next, we set the likelihood to be of "Direct" type
```python
k["Bayesian"]["Likelihood"]["Type"] = "Direct";

k.setLikelihood( evaluateModel )
```

###  The Variables
In this problem there is only one variable,
```python
k["Variables"][0]["Name"] = "X";
```

Now we have to define a uniform prior for the parameter
```python
k["Variables"][0]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
k["Variables"][0]["Bayesian"]["Prior Distribution"]["Minimum"] = -10.0;
k["Variables"][0]["Bayesian"]["Prior Distribution"]["Maximum"] = +10.0;
```



###  The Solver
We choose the solver `MCMC` and set some parameters for TMCMC.
```python
k["Solver"] = "TMCMC";
k["TMCMC"]["Covariance Scaling"] = 0.02;
k["TMCMC"]["Population Size"] = 5000;
k["TMCMC"]["Min Rho Update"] = 0.0;
k["TMCMC"]["Coefficient of Variation"] = 0.5;
k["TMCMC"]["Burn In"] = 5
```

 For a detailed description of the MCMC settings see [here](../../usage/solvers/samplers/tmcmc.md).

Finally, we need to add a call to the run() routine to start the Korali engine.

```python
k.run()
```

###  Run

We are now ready to run our example:

```bash
./a2-sampling-mcmc
```

Or, alternatively:

```bash
python3 ./a2-sampling-mcmc
```

The results are saved in the folder `_korali_result/`.


###  Plot
You can see a histogram of the results by running the command
```sh
python3 -m korali.plotter
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

    To use `jsondecode` Matlab version must be >= R2016b.

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
