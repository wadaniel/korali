Model Propagation (Execution)
=====================================================

In this tutorial we show how to pass multiple inputs through a model.


## Problem Setup

We are given a set of parameters to evaluate in a file 'samplesOut.dat'.
We want to execute a model function $f(\theta)$ on given parameters.

## Optimization / Sampling  with TODO

###  The Korali Object


```python
e = korali.Experiment()
```
 
###  The Problem Type
The type of problem is `Execution/Model`.
```python
e["Problem"]["Type"] = "Execution/Model"
```

###  The Variables

In the file we have means and variances to evaluate:

```python
e["Variables"][0]["Name"] = "Mean"
e["Variables"][0]["Loaded Values"] = means
e["Variables"][1]["Name"] = "Variance"
e["Variables"][1]["Loaded Values"] = variances
```

###  The Solver
We set the solver and choose how many samples are evaluated per generation.


```python
e["Solver"]["Type"] = "Executor"
e["Solver"]["Executions Per Generation"] = 1
```

###  Running

We are now ready to run our example:

```bash
./a6-executionpy
```

Or, alternatively:

```bash
python3 a6-execution.py
```

The results are saved in the folder `_korali_result/`.
