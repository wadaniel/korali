# Test: UNIT-002

Test MPI Conduit with Bayesian Inference

## Description

Tests the distributed MPI conduit for a bayesian inference problem using a heat diffusion solver on 2D. 

## Source

[https://github.com/cselab/korali/tree/master/tests/UNIT-002](https://github.com/cselab/korali/tree/master/tests/UNIT-002)

## Steps

### Step 1

+ Operation: Compile cpp files.
+ Expected: Runs without errors and rc = 0.  If MPI is not installed, it will not compile.

### Step 2

+ Operation: Run 4x2 distribution. 
+ Expected: 4 Concurrent teams of 2 MPI ranks run with rc = 0. If MPI is not installed, it will not run.

### Step 3

+ Operation: Run 1x8 distribution. 
+ Expected: 1 team of 8 MPI ranks run with rc = 0. If MPI is not installed, it will not run.

