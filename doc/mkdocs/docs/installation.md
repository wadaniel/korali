# Installation

## Steps

### 1. Check your system

 Check the [Requirements](#system-requirements) section to make sure your system provides
 the required compilation and software requirements for Korali.

### 2. Download Korali
Download the Korali project code from [GitHub](https://github.com/cselab/skorali)

```shell
git clone https://github.com/cselab/skorali
```

### 3. Build and Install Korali

To build and install Korali, run:

```shell
./install
```

!!! note
    Use ```./install --install-gsl``` to have Korali download and build the GSL-2.5 GNU Scientific Library automatically. 
	

## System Requirements

### C++ Compiler

Korali requires a C++ that supports the C++14 standard (`-std=c++14`) to build its engine, dependencies, and user applications.

We have successfully tested Korali in the following environments:

  - g++ (GCC) 6.3.0, CentOS Linux 7
  - g++ (GCC) 8.2.0, CentOS Linux 7
  - Apple LLVM version 10.0.1 (clang-1001.0.46.4), macOS 10.14.4 (Darwin 18.5.0)
  - Cray Programming Environment 2.5, SUSE Linux Enterprise Server 12 SP3
  
  You can verify your C++ compiler and OS versions with the following commands:

```shell
# For Linux
g++ --version
cat /etc/os-release

# For MacOS
clang++ --version
system_profiler SPSoftwareDataType
```

Korali's installer will check the $CXX environment variable to determine the default C++ compiler. You can change the value of this variable to define a custom C++ compiler.

### Python >3.0

Korali requires a version of Python higher than 3.0 to run Python models and plotting their results.

We have successfully tested Korali with the following Python versions:

  - Python 3.3.0
  - Python 3.7.0
  - Python 3.7.1

Korali's installer will check the 'python3' command. The path to this command should be present in the $PATH environment variable. Make sure your Python is correctly installed or its module loaded before configuring Korali.


## Optional Requirements

### Git

The Git control version software is required in case you choose to download Korali's code from our repository or
wish to contribute with its development. You can download git [here](https://git-scm.com/downloads).

### UPC++ v2018.9.0

To enable Korali's UPC++ conduit for distributed memory sampling, you first need a valid installation of the [UPC++ library](https://bitbucket.org/berkeleylab/upcxx/wiki/Home) v2018.9.0.
In addition, you need to explicitly require Korali to enable the UPC++ conduit, through:

```shell
./install --enable-upcxx
```

Korali's installer will check the 'upcxx' command for support for UPC++. The path to this command should be present in the $PATH environment variable. Once Korali finds its path, it will check for UPC++'s compilation and
linking flags with:

```shell
upcxx-meta CPPFLAGS
upcxx-meta LIBS
```

To download and install UPC++, you can follow these instructions:

```shell
wget https://bitbucket.org/berkeleylab/upcxx/downloads/upcxx-2018.9.0.tar.gz
tar -xzvf upcxx-2018.9.0
cd upcxx-2018.9.0
./install $HOME/myapps/upcxx
export PATH=$PATH:$HOME/myapps/upcxx/bin
```

### MPI

Korali's installer will check the $MPICXX environment variable to determine a valid MPI C++ compiler. To enable Korali's MPI conduit for distributed memory sampling, you first need a valid installation of the MPI library or make sure you have the corresponding modules loaded in your system.

### GNU Scientific Library

Korali requires the [GSL-2.4](http://www.gnu.org/software/gsl/) or later must be installed on your system. Korali will look for the ```gsl-config``` command by default. In case GSL is not currently installed, you can run ```./install --install-gsl``` to have Korali install and build it automatically.


