

In this tutorial we show how to solve a **constrained optimization** problem ([CEC 2006](http://web.mysites.ntu.edu.sg/epnsugan/PublicSite/Shared%20Documents/CEC-2006/technical_report.pdf) Test Problem g09), defined as: Find $x^\star = \mathop{\arg\min}\limits_{x}  \,\,\, f(x) \,,$ under the constraints $g_i(x) \leq 0.$


!!! info
	An example Python code for this tutorial can be found in:
	[tutorials/python/b2-constrained-optimization.py](https://github.com/cselab/skorali/blob/master/tutorials/python/b2-constrained-optimization.py)
	
	An example C++ code for this tutorial can be found in:
	[tutorials/cxx/b2-constrained-optimization.cpp](https://github.com/cselab/skorali/blob/master/tutorials/cxx/b2-constrained-optimization.cpp)


## Problem Setup

We want to solve the problem:

$$ f(x) = (x_1 - 10)^2 + 5(x_2 - 12)^2 + x_3^4  + 3(x_4 - 11)^2 + 10 x_5^6 + \\
7 x_6^2 + x7^4 - 4 x_6 x_7 - 10  x_6 - 8 x_7 $$

under the four constraints $g_i(x)$:

$$
g_1(x) = -127 + 2x_1^2 + 3 x_2^4 + x_3 + 4x_4^2 + 5x_5
$$

$$
g_2(x) = -282 + 7x_1 + 3x_2 + 10x_3^2 + x_4 - x_5
$$

$$
g_3(x) = -196 + 23 x_1 + x_2^2 + 6x_6^2 - 8x_7
$$

$$
g_4(x) = 4x_1^2 + x_2^2 - 3x_1x_2 + 2x_3^2 + 5x_6 - 11x_7
$$


##  The Objective Function

Create a folder named `model`. Inside, create a file with name `g09.py` and paste the following code,
```python
#!/usr/bin/env python

def g09( s ):

  nPars = s.getVariableCount();
  if( nPars != 7 ) :
    print("Error in g09: Number of variables must be 7.\n");
    exit();


  pars = s.getVariables();

  res = (pars[0] - 10.0)**2 + 5.0 * (pars[1] - 12.0)**2           \
        + pars[2]**4  + 3.0 * (pars[3] - 11.0)**2                 \
        + 10.0 * pars[4]**6 + 7.0 * pars[5]**2 + pars[6]**4.      \
        - 4.0 * pars[5] * pars[6] - 10.0 * pars[5] - 8.0 * pars[6];

  s.addResult(-res);
```
This computational model represents our objective function.

For the constraints, add the following code in the same file,
```python
def g1( x ):
  return  -127.0 + 2 * x[0] * x[0] + 3.0 * pow(x[1], 4) + x[2] \
          + 4.0 * x[3] * x[3] + 5.0 * x[4];


def g2( x ):
  return -282.0 + 7.0 * x[0] + 3.0 * x[1] + 10.0 * x[2] * x[2] \
          + x[3] - x[4];


def g3( x ):
  return -196.0 + 23.0 * x[0] + x[1] * x[1] + 6.0 * x[5] * x[5] \
          - 8.0 * x[6];

def g4( x ):
  return  4.0 * x[0] * x[0] + x[1] * x[1] - 3.0 * x[0] * x[1] \
          + 2.0 * x[2] * x[2] + 5.0 * x[5] - 11.0 * x[6];
```

## Optimization with CCMA-ES

First, open a file and import the korali module
```python
#!/usr/bin/env python3
import korali
```
Import the computational model,
```python
import sys
sys.path.append('./model')
from g09 import *
```

###  The Korali Object

Next we construct a `Korali` object and set the computational model,
```python
k = korali.initialize()
```

Add the objective function and the constraints in the Korali object,
```python
k.setModel( g09 );
k.addConstraint( g1 );
k.addConstraint( g2 );
k.addConstraint( g3 );
k.addConstraint( g4 );
```


###  The Problem Type
Then, we set the type of the problem to `Direct Evaluation`
```python
k["Problem"] = "Direct Evaluation"
```


###  The Variables
We add 7 variables to Korali,
```python
nParams = 7;
for i in range(nParams) :
  k["Variables"][i]["Name"] = "X" + str(i);
```


###  The Solver
We choose the solver `CMA-ES` and the domain of the parameter `X`,

```python
k["Solver"]  = "CCMA-ES";

for i in range(nParams) :
  k["Variables"][i]["CCMA-ES"]["Lower Bound"] = -10.0;
  k["Variables"][i]["CCMA-ES"]["Upper Bound"] = +10.0;
```

Then we set a few parameters for CCMA-ES,
```python
k["CCMA-ES"]["Adaption Size"] = 0.1;
k["CCMA-ES"]["Sample Count"] = 8;
k["CCMA-ES"]["Viability Sample Count"] = 2;
k["CCMA-ES"]["Termination Criteria"]["Min Fitness"]["Value"] = -680.630057374402 - 1e-4;
```
For a detailed description of CCMA-ES settings see [here](../../usage/solvers/optimizers/ccmaes.md).

Finally, we need to add a call to the run() routine to start the Korali engine.

```python
k.run()
```

###  Running

We are now ready to run our example:

```bash
./b2-constrained-optimization
```

Or, alternatively:

```bash
python3 ./b2-constrained-optimization
```

The results are saved in the folder `_korali_result/`.

###  Plotting

You can see the results of CMA-ES by running the command,
```sh
python3 -m korali.plotter
```

![figure](ccmaes.png)
