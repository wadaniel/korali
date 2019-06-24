# Integration

!!!warning
	This page is just a demonstration of how continuous integration will work

Current Status:  <span style="color:green">All Tests Passed</span> 

## Regression Tests

|  Status  |   Time      |  Commit     |   System      |  Description     |
| -------- |:-------------:|: ----------------:|: ----------------:|: ----------------:|
| <span style="color:green">Passed</span> | 20/06/2019 15:58 | ec18366 | falcon.ethz.ch  | g++ (GCC) 6.3.0, CentOS Linux 7        |
| <span style="color:green">Passed</span> | 20/06/2019 15:58 | ec18366 | pachnes.ethz.ch | Apple LLVM version 10.0.1 (clang-1001.0.46.4), macOS 10.14.4 (Darwin 18.5.0)      |
| <span style="color:green">Passed</span> | 20/06/2019 15:58 | ec18366 | daint.cscs.ch   | Cray PE 2.5, SUSE Linux Enterprise Server 12 SP3      |

## Unit Tests

|  Status  |   Time      |  Commit     |   Test Code   |  Description     |
| -------- |:-------------:|: ----------------:|: ----------------:|: ----------------:|
| <span style="color:green">Passed</span> | 20/06/2019 15:58 | ec18366 | [A001](/development/unit/a001.html) | CMA-ES stress Tests |
| <span style="color:green">Passed</span> | 20/06/2019 15:58 | ec18366 | [A002](/development/unit/a002.html) | TMCMC stress Tests  |
| <span style="color:green">Passed</span> | 20/06/2019 15:58 | ec18366 | [A003](/development/unit/a003.html) | Nonintrusive Conduit Tests    |
| <span style="color:green">Passed</span> | 20/06/2019 15:58 | ec18366 | [A004](/development/unit/a004.html) | Bayesian Inference Tests   |
| <span style="color:green">Passed</span> | 20/06/2019 15:58 | ec18366 | [A005](/development/unit/a005.html) | MPI Conduit |