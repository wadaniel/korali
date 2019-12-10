# Test: STAT--003

Check Sampling Statistics

## Description

This test checks the minimization outcome of DEA and CMA-ES to a problem that
returns Infs and Nans, correct execution under this conditions will be tested.

## Source

[https://github.com/cselab/korali/tree/master/tests/STAT-003](https://github.com/cselab/korali/tree/master/tests/STAT-003)

## Steps

### Step 1

+ Operation: Execute run-dea.py and run-mcmc.py
+ Expected: The defaults are correct, and rc = 0.

### Step 2

+ Operation: Check correctness of minimum obtained.
+ Expected: The defaults are correct, and rc = 0.
