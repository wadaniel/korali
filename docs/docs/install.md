# Installation

## Instructions

- **1. Download Korali**

Download the latest stable release of Korali with the following command:

```shell
git clone https://github.com/cselab/korali.git
```

- **2. Install Korali**

To build and install Korali, simply run:

```shell
cd korali
./install
```

??? question "Trouble Installing Korali?"
	If you are experiencing problems installing or running Korali, please check the following hints:

	- Visit the [system requirements](../requirements) page to verify that your system meets the software requirements to install and run Korali.

	- Check the [build](#build-status) to see if Korali is currently building correctly.

	- If the problem persists, please submit a new [issue report](https://github.com/cselab/korali/issues) on our Github repository detailing the problem, your system information, and the steps to replicate it and we will promptly address it.

	- For further questions, feel free to [contact us](aboutus).

## System Requirements

??? info "Mandatory Requirements"
	+ **C++ Compiler**
    	Korali requires a C++ compiler that supports the C++14 standard (`-std=c++14`) to build.
    	**Hint:** Check the following [resource](https://en.cppreference.com/w/cpp/compiler_support#C.2B.2B14_core_language_features) to verify whether your compiler supports C++14.
    	Korali's installer will check the **$CXX** environment variable to determine the default C++ compiler. You can change the value of this variable to define a custom C++ compiler.
  
	+ **GNU Make**
    	Korali requires GNU Make to build its C++ engine.
  
	+ **wget**
    	Korali requires access to the internet and the *wget* command to be available to automatically resolve some of its dependencies. If you wish to install Korali on a system without access to internet or no *wget* command, you check the [Installed by Korali](#installed-by-korali) section of this document to manually define these requirements.
  
	+ **Git Client**
	You need Git to clone (download) our code before installation.
  
	+ **Python3**
	Korali requires a version of Python higher than 3.0 to be installed in the system. Korali's installer will check the *python3* command. The path to this command should be present in the $PATH environment variable. *Hint:* Make sure Python3 is correctly installed or its module loaded before configuring Korali.
  
	+ **python3-config**
	Korali requires the command *python3-config* to be available during installation. This command is typically included in standard installations of python3 that include developer tools. *Hint:*  If *python3-config* is missing, you can get it by installing/loading the **python3-dev** package/module in your system.
  
	+ **Pip3 Installer**
	Korali requires the *pip3* command to install it's engine and tools. This command is typically included in standard installations of python. *Hint:*  If *pip3* is missing, you can get it by installing pip3, with e.g, `brew install pip3`
  
	** Installed Automatically by Korali: **
  
	+ **PyBind11**
	Korali requires *pybind11* to enable Python/C++ interaction. If not found, it will try to install it automatically using *pip3*.
  
	+ **GNU Scientific Library**
	Korali requires the [GSL-2.4](http://www.gnu.org/software/gsl/) or later must be installed on your system. If the command ```gsl-config``` is not found, Korali will try to install GSL automatically. *Hint:* If you do not want Korali downloading and installing GSL on its own, run './install --no-prereqs'

??? info "Optional Requirements"
	+ **MPI Library**
	One way to enable support distributed conduits and computational models is to configure Korali to compile with an MPI compiler. The installer will check the *$MPICXX* environment variable to determine a valid MPI C++ compiler.
  
	+ **UPC++ Library**
	One way to enable support distributed conduits and computational models is to configure Korali to compile with an UPC++ compiler. The installer will check the *upcxx-meta* command avaiable through $PATH to determine a valid UPC++ compiler.

## Build Status

| Status         |     Branch             |    URL         | 
| :----------------:|:----------------:|: ----------------- :|
| [![CircleCI](https://circleci.com/gh/cselab/korali/tree/master.svg?style=svg&circle-token=d73f56a4d14073880f8fe1140964afb58f2b1c35)](https://circleci.com/gh/cselab/korali/tree/master) | master | [https://github.com/cselab/korali/tree/master](https://github.com/cselab/korali/tree/master) | 
| [![CircleCI](https://circleci.com/gh/cselab/korali/tree/development.svg?style=svg&circle-token=d73f56a4d14073880f8fe1140964afb58f2b1c35)](https://circleci.com/gh/cselab/korali/tree/development) | development | [https://github.com/cselab/korali/tree/development](https://github.com/cselab/korali/tree/development) | 

??? info "Test Architectures"

    |     System           |      Compiler     |    Python         |
    | :--------------------:|: ----------------:|: ----------------:|
    | [Debian GNU/Linux 9](https://github.com/CircleCI-Public/circleci-dockerfiles/tree/master/python/images/3.7.3) | gcc version 6.3.0 | Python 3.7.3  |
    | [macOS 10.13.6  (Darwin 17.7.0)](https://circle-macos-docs.s3.amazonaws.com/image-manifest/build-456/index.html) | Apple LLVM version 10.0.1 (clang-1001.0.46.4) | Python 3.7.3     |
  
??? info "Test Collection"
  
    | Type             |     Code         |     Description    | 
    | :----------------:|:----------------:|: ----------------:|
    | Regression | [REG-000](https://github.com/cselab/korali/tree/development/tests/REG-000/run_test.sh) | Checks for a correct installation of Korali and its modules. |
    | Regression | [REG-001](https://github.com/cselab/korali/tree/development/tests/REG-001/run_test.sh) | Re-run all Python tutorials for basic sanity check. |
    | Regression | [REG-002](https://github.com/cselab/korali/tree/development/tests/REG-002/run_test.sh) | Re-run all C++ tutorials for basic sanity check. |
    | Regression | [REG-003](https://github.com/cselab/korali/tree/development/tests/REG-003/run_test.sh) | Compare python and cxx runs to ensure they produce same results. |
    | Regression | [REG-004](https://github.com/cselab/korali/tree/development/tests/REG-004/run_test.sh) | Test correct plotting for all results from tutorial runs. |
    | Regression | [REG-005](https://github.com/cselab/korali/tree/development/tests/REG-005/run_test.sh) | Perform dry runs of all experiments in the korali-apps repository. |
    | Statistical | [STAT-000](https://github.com/cselab/korali/tree/development/tests/STAT-000/run_test.sh) | Check that solvers initialize with correct default values. |
    | Statistical | [STAT-001](https://github.com/cselab/korali/tree/development/tests/STAT-001/run_test.sh) | Check Sampling Statistics |
    | Unit Test | [UNIT-000](https://github.com/cselab/korali/tree/development/tests/UNIT-000/run_test.sh) | Test for the External Conduit |
    | Unit Test | [UNIT-001](https://github.com/cselab/korali/tree/development/tests/UNIT-001/run_test.sh) | Parallel Linked Conduit for Sequential Bayesian Inference |
    | Unit Test | [UNIT-002](https://github.com/cselab/korali/tree/development/tests/UNIT-002/run_test.sh) | Parallel Linked Conduit for Distributed (MPI) Bayesian Inference |

