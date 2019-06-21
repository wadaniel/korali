

In this tutorial we show how to **optimize** directly from a
given function.

The code in **Python** for this tutorial can be found [here](https://github.com/cselab/skorali/blob/master/tutorials/a1-optimization).



## Problem setup
We are given the function $g(\vartheta)=\exp(-\vartheta^2)$ for $\vartheta\in[-10,10]$.
We want to find the maximum of the function in the given interval.

!!! tip
    Since $\exp$ is monotonically increasing function, g has the same optima
    as $\log g$.



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
We choose the solver `CMA-ES`, set the domain of the parameter `X`, the population size to be `5` and two termination criteria,

```python
k["Solver"]  = "CMA-ES"

k["Variables"][0]["CMA-ES"]["Lower Bound"] = -10.0;
k["Variables"][0]["CMA-ES"]["Upper Bound"] = +10.0;

k["CMA-ES"]["Objective"] = "Maximize"
k["CMA-ES"]["Termination Criteria"]["Max Generations"]["Value"] = 500
k["CMA-ES"]["Sample Count"] = 5
```
For a detailed description of CMA-ES settings see [here](../../usage/solvers/optimizers/cmaes.md).



###  Run

Finally, we are ready to run the simulation,

```python
k.run()
```

The results are saved in the folder `_korali_result/`.


###  Plot

You can see the results of CMA-ES by running the command,
```sh
python3 -m korali.plotter
```

![figure](direct-cma.png)
