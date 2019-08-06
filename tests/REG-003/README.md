# Test: REG-003

Test correct configuration of real-world Korali-based applications.

## Description

This test configures (without running) all the applications in the korali-apps repository to make sure they are up-to-date with Korali's syntax.

## Source

[https://github.com/cselab/korali/tree/master/tests/REG-003](https://github.com/cselab/korali/tree/master/tests/REG-003)

## Steps

### Step 1

+ Operation: Clone the korali-apps repository
+ Expected: No errors (rc = 0).

### Step 2

+ Operation: Dry Run the Korali+LAMMPS Application
+ Expected: No errors (rc = 0).

### Step 3

+ Operation: Dry Run the Korali+Mirheo Application
+ Expected: No errors (rc = 0)

### Step 4

+ Operation: Dry Run the Korali+MSOLVE Application
+ Expected: No errors (rc = 0)