# Installation

## Installing Korali

### 1. Check your system

 Check the Requirements section to make sure your system provides
 the required compilation and software requirements for Korali.

### 2. Download Korali
Download the Korali project from [GitHub](https://github.com/cselab/skorali)

```shell
git clone https://github.com/cselab/skorali
```

tar -xzvf

### 3. Run the configuration script

Check the full syntax for configure here.

```shell
./configure
```
By default the install directory is `./korali`. If you want to install in a different directory, run
```shell
./configure --prefix=/path/to/your/directory
```

### 4. Build and Install Korali

Finally, install Korali
```shell
make install
```
In the beginning the GSL-2.5 GNU Scientific Library will be downloaded and installed in your system. This will take some time.

### [Recommended] 5. Add Path and Python Path folders

**Don't forget to add the binary and library folders to the path!**

### [Optional] 6. Run tests

To make sure Korali runs correctly.

## Prerequisities

### C++ Compiler

### Python >3.0

### [Optional] Git (for download)

### [Optional] UPC++
Download UPC++
```shell
wget https://bitbucket.org/berkeleylab/upcxx/downloads/upcxx-2019.3.0.tar.gz
```
and install in a desired directory
```shell
./install /path/to/your/directory
```

- [UPCXX](https://bitbucket.org/berkeleylab/upcxx/wiki/Home) v2019.3.0.

### [Optional] MPI
- MPI implementation preferably with full thread safety, e.g. [MPICH](http://www.mpich.org).


### [Optional] GSL
- [GSL-2.4](http://www.gnu.org/software/gsl/) or later must be installed on your system.


