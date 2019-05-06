## Problem setup
The problem setup is the same as in [this example](../cxx/posterior.md).

The code for this tutorial can be found [here](https://github.com/cselab/skorali/blob/master/examples/cxx/quick_start/posterior.python).



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
