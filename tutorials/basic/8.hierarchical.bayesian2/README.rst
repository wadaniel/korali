# A.8 - Simple Hierarchical Bayesian Inference

In this tutorial we show how to do hierarchical Bayesian inference.


## Problem Setup

Hierarchical Bayesian Inference is set up in 3 phases:
 - sample the posterior distributions conditioned on each data set
 - sample the hyper parameter
 - sample the posterior given hyperparameter and one (a) data set or  (b) data sets combined.

For each phase we provided an individual `python` file.

In phase 0 we generate synthetic data.

###  Running

All 3 phases can be run with the shell script

```bash
./run-hierarchical.sh
```

The results are saved in sub dirs of the folder `/setup`.
