## Problem setup
In this section we will present how to optimize or sample from a given function $f(\vartheta)=\vartheta^2$ for $\vartheta\in[-10,10]$.

The code for this tutorial can be found [here](https://github.com/cselab/skorali/blob/master/examples/python/quick_start/direct.python).

## Optimize

First import the needed modules
```python
#!/usr/bin/env python
import sys
import threading
import libkorali
```

### 1. The computational model
Then write the code for $f$
```python
def F( s ):
    x = s.getParameter(0)
    r = -(x-2.)**2
    s.addResult(r)
```


### 2. The korali object

Next we construct a `korali` object using the lambda function
```python
korali = libkorali.Engine( F )
```

### 3. The Problem type
The we set the type of the problem to `Direct Evaluation`
```python
korali["Problem"]["Objective"] = "Direct Evaluation"
```

### 4. The Parameters
```python
korali["Parameters"][0]["Name"] = "X"
korali["Parameters"][0]["Distribution"] = "Uniform"
korali["Parameters"][0]["Type"] = "Computational"
korali["Parameters"][0]["Minimum"] = -10.0
korali["Parameters"][0]["Maximum"] = +10.0
```

### 5. The Solver
```python
korali["Solver"]["Method"] = "CMA-ES"
korali["Solver"]["Termination Criteria"]["Min DeltaX"] = 1e-8
korali["Solver"]["Termination Criteria"]["Max Generations"] = 100
korali["Solver"]["Lambda"] = 5
```

### 6. Run
```python
korali["Seed"] = 0xC0FFEE;
korali["Verbosity"] = "Detailed";
```


```python
korali.run();
```






## Sample

### 1. The Solver

```python
korali["Solver"]["Method"] = "TMCMC"
korali["Solver"]["Covariance Scaling"] = 0.02
korali["Solver"]["Population Size"] = 5000
korali["Solver"]["Burn In"] = 5
```
