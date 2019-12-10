# A.5 - Model Optimization with Discrete (Integer) Variables

In this tutorial we show how to optimize a function with discrete input parameter.


## Problem Setup

##  The Objective Function

## Optimization with CCMA-ES

###  The Korali Object

###  The Problem Type

###  The Variables

###  The Solver
We choose the solver `CMA-ES` and the domain of the parameter `X`,

###  Running

We are now ready to run our example:

```bash
./a5-constrained-optimization
```

Or, alternatively:

```bash
python3 ./a5-constrained-optimization
```

The results are saved in the folder `_korali_result/`.

###  Plotting

You can see the results of CMA-ES by running the command,
```sh
python3 -m korali.plotter
```

