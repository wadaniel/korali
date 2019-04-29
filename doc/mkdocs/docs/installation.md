# Installation

## Prerequisities

### 1. Download UPC++
Download UPC++
```shell
wget https://bitbucket.org/berkeleylab/upcxx/downloads/upcxx-2019.3.0.tar.gz
```
and install in a desired directory
```shell
./install /path/to/your/directory
```


## Installation

### 1. Download Korali
Download the Korali project from [GitHub](https://github.com/cselab/skorali)

```shell
git clone https://github.com/cselab/skorali
```


### 2. Install Korali
First, set the desired C++ compiler

```shell
 export CXX=gcc
```
The compiler must support...

Then configure

```shell
./configure
```

By default the install directory is `./korali`. If you want to install in a different directory, run
```shell
./configure --prefix=/path/to/your/directory
```

Finally, install Korali
```shell
make install
```
In the beginning the GSL-2.5 GNU Scientific Library will be downloaded and installed in your system. This will take some time.

Add the executable and library to the path,
```shell
export PATH=$PATH:/home/garampat/skorali/korali/bin
export PYTHONPATH=$PYTHONPATH:/home/garampat/skorali/korali/lib
```
