# Build Status

<center>

## Branches

| Status         |     Branch             |    URL         | 
| :----------------:|:----------------:|: ----------------- :|
| [![CircleCI](https://circleci.com/gh/cselab/korali/tree/master.svg?style=svg&circle-token=d73f56a4d14073880f8fe1140964afb58f2b1c35)](https://circleci.com/gh/cselab/korali/tree/master) | master | [https://github.com/cselab/korali/tree/master](https://github.com/cselab/korali/tree/master) | 
| [![CircleCI](https://circleci.com/gh/cselab/korali/tree/development.svg?style=svg&circle-token=d73f56a4d14073880f8fe1140964afb58f2b1c35)](https://circleci.com/gh/cselab/korali/tree/development) | development | [https://github.com/cselab/korali/tree/development](https://github.com/cselab/korali/tree/development) | 

## Test Architectures

|     System           |      Compiler     |    Python         |
| :--------------------:|: ----------------:|: ----------------:|
| [Debian GNU/Linux 9](https://github.com/CircleCI-Public/circleci-dockerfiles/tree/master/python/images/3.7.3) | gcc version 6.3.0 | Python 3.7.3  |
| [macOS 10.13.6  (Darwin 17.7.0)](https://circle-macos-docs.s3.amazonaws.com/image-manifest/build-456/index.html) | Apple LLVM version 10.0.1 (clang-1001.0.46.4) | Python 3.7.3     |

## Test Collection

| Type             |     Code         |     Description    | 
| :----------------:|:----------------:|: ----------------:|
| Regression | [REG-000](https://github.com/cselab/korali/tree/development/tests/REG-000/run_test.sh) | Checks for a correct installation of Korali and its modules. |
| Regression | [REG-001](https://github.com/cselab/korali/tree/development/tests/REG-002/run_test.sh) | Check that solvers initialize with correct default values. |
| Regression | [REG-002](https://github.com/cselab/korali/tree/development/tests/REG-003/run_test.sh) | Re-run all tutorials for basic sanity check. |
| Regression | [REG-003](https://github.com/cselab/korali/tree/development/tests/REG-004/run_test.sh) | Compare python and cxx runs to ensure they produce same results. |
| Regression | [REG-004](https://github.com/cselab/korali/tree/development/tests/REG-004/run_test.sh) | Test correct plotting for all results from tutorial runs. |
| Unit Test | [UNIT-000](https://github.com/cselab/korali/tree/development/tests/UNIT-000/run_test.sh) | Test for the External Conduit |
| Unit Test | [UNIT-001](https://github.com/cselab/korali/tree/development/tests/UNIT-001/run_test.sh) | Parallel Linked Conduit for Sequential Bayesian Inference |
| Unit Test | [UNIT-002](https://github.com/cselab/korali/tree/development/tests/UNIT-002/run_test.sh) | Parallel Linked Conduit for Distributed (MPI) Bayesian Inference |

</center>
