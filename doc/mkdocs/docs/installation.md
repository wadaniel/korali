# Installation

## Prerequisities
- [UPCXX](https://bitbucket.org/berkeleylab/upcxx/wiki/Home) v2019.3.0.
- MPI implementation preferably with full thread safety, e.g. [MPICH](http://www.mpich.org).
- [GSL-2.4](http://www.gnu.org/software/gsl/) or later must be installed on your system.


## Installation steps


### 0. Download
Download the Korali project from [GitHub](https://github.com/cselab/skorali).


### 1. GSL Library

The [GNU Scientific Library (GSL)](https://www.gnu.org/software/gsl/) is a numerical library for C and C++ programmers.

- Download and unpack the latest version of GSL (outside of korali). Execute following commands in your terminal:
```sh
	wget http://mirror.easyname.at/gnu/gsl/gsl-latest.tar.gz  
	tar -xvzf gsl-latest.tar.gz
```

- Configure GSL. Set the install folder to be `/$HOME/usr`. If you want to install in the default diretcory, `/usr/local`, delete the  `--prefix=/$HOME/usr`.
```sh
	./configure   --prefix=/$HOME/usr
```

- Compile and install GSL.
```sh
	make -j2
	make install
```
This step will take some time. If your machine has more than two cores, set the 2 in the -j2 to that number number.

#### GSL Installation Hints

- Read the INSTALL notes of GSL.
- Update the path variable of your terminal with the location of the binaries (gsl-config, gsl-histogram and gsl-randist).
- Set the LD_LIBRARY_PATH variable of your terminal to the newly installed lib folder.
 


### 2. Korali Installation

Before compiling, the following needs to be checked:

- PREFIX (install dir) in `skorali/common.mk`.  

**Compilation options:**  

TODO



