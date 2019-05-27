# System Requirements

## Mandatory Requirements

### C++ Compiler

Korali requires a C++ that supports the C++17 standard (`-std=c++17`) to build its engine, dependencies, and user applications.

We have successfully tested Korali in the following environments:

  - g++ (GCC) 6.3.0, CentOS Linux 7
  - g++ (GCC) 8.2.0, CentOS Linux 7
  - Apple LLVM version 10.0.1 (clang-1001.0.46.4), macOS 10.14.4 (Darwin 18.5.0)

You can verify your C++ compiler and OS versions with the following commands:

```shell
# For Linux
g++ --version
cat /etc/os-release

# For MacOS
clang++ --version
system_profiler SPSoftwareDataType
```

By default, Korali's [configuration](./configuration.md) script will check the $CXX environment variable to determine
the default compiler. You can change the value of this variable to define a custom C++ compiler.

### Python >3.0

Korali requires a version of Python higher than 3.0 to run Python models and plotting their results.

We have successfully tested Korali with the following Python versions:

  - Python 3.3.0
  - Python 3.7.0
  - Python 3.7.1

By default, Korali's [configuration](./configuration.md) script will check the 'python3' command. The path to this command
should be present in the $PATH environment variable. Once Korali finds its path, it will check for its corresponding pkg-config
information in:

```shell
pythonbin=`which python3`
pythonpath=`dirname $pythonbin`
$pythonpath/../lib*/pkgconfig/python3.pc'.
```

Make sure your version of python is correctly installed and the .pc file exists before configuring Korali.


## Optional Requirements

### Git

The Git control version software is required in case you choose to download Korali's code from our repository or
wish to contribute with its development. You can download git [here](https://git-scm.com/downloads).

### UPC++ v2018.9.0

To enable Korali's UPC++ conduit for distributed memory sampling, you first need a valid installation of the [UPC++ library](https://bitbucket.org/berkeleylab/upcxx/wiki/Home) v2018.9.0.
In addition, you need to explicitly require Korali to enable the UPC++ conduit, through:

```shell
./configure --enable-upcxx
```

By default, Korali's [configuration](./configuration.md) script will check the 'upcxx' command. The path to this command
should be present in the $PATH environment variable. Once Korali finds its path, it will check for UPC++'s compilation and
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

To enable Korali's MPI conduit for distributed memory sampling, you first need a valid installation of the MPI library.

MPI implementation preferably with full thread safety, e.g. [MPICH](http://www.mpich.org).


### GNU Scientific Library

Korali requires the [GSL-2.4](http://www.gnu.org/software/gsl/) or later must be installed on your system.

Korali will, by default, download and install the GNU Scientific Library (GSL) during its build. If you have a version
of GSL that you would prefer to use instead of the default, you can specify through the following option:

```shell
./configure --use-gsl=PATH_TO_GSL
```
