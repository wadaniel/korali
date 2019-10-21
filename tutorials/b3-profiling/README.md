# Test: UNIT-000

Test for the External Conduit 

## Description

Simple test that runs 30 generations of CMAES running the Ackley function that is executed as a standalone process

## Source

[https://github.com/cselab/korali/tree/master/tests/UNIT-000](https://github.com/cselab/korali/tree/master/tests/UNIT-000)

## Steps

### Step 1

+ Operation: Run external.py with 1 process
+ Expected: Runs without errors and rc = 0.

### Step 2

+ Operation: Run external.py with 4 concurrent processes.
+ Expected: Runs without errors and rc = 0.

### Step 3

+ Operation: Run external.py with 8 concurrent processes.
+ Expected: Runs without errors and rc = 0.
