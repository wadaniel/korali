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
cd skorali
./install
```

## System Requirements

### C++ Compiler

Korali requires a C++ that supports the C++14 standard (`-std=c++14`) to build its engine, dependencies, and user applications.

!!! note
		The installer will check the $CXX environment variable to determine the default C++ compiler. You can change the value of this variable to define a custom C++ compiler.

### Python >3.0

Korali requires a version of Python higher than 3.0 to be installed in the system.

!!! note
	Korali's installer will check the 'python3' command. The path to this command should be present in the $PATH environment variable. Make sure your Python is correctly installed or its module loaded before configuring Korali.

### GNU Scientific Library

Korali requires the [GSL-2.4](http://www.gnu.org/software/gsl/) or later must be installed on your system.

!!! hint
	You can use ```./install --install-gsl``` to have Korali download and build the GSL-2.5 GNU Scientific Library automatically. 
	
### MPI (Optional)

To enable support distributed conduits and computational models, the MPI library should be installed in the system.

!!! note
	The installer will check the $MPICXX environment variable to determine a valid MPI C++ compiler. Also make sure you have the corresponding modules loaded in your system.

