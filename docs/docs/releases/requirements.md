## System Requirements

### C++ Compiler

Korali requires a C++ that supports the C++14 standard (`-std=c++14`) to build its engine, dependencies, and user applications.

!!! note
		The installer will check the $CXX environment variable to determine the default C++ compiler. You can change the value of this variable to define a custom C++ compiler.

### Python >3.0

Korali requires a version of Python higher than 3.0 to be installed in the system.

!!! note
	Korali's installer will check the 'python3' command. The path to this command should be present in the $PATH environment variable. Make sure your Python is correctly installed or its module loaded before configuring Korali.

### Pip3 Installer

Korali requires pip3 to be installed in the system.

### PyBind11 

Korali requires pybind11 to be installed in the system. If not found, it will try to install it automatically using pip3.


### GNU Scientific Library

Korali requires the [GSL-2.4](http://www.gnu.org/software/gsl/) or later must be installed on your system. If the command ```gsl-config``` is not found, Korali will try to install GSL automatically.
	
### MPI (Optional)

To enable support distributed conduits and computational models, the MPI library should be installed in the system.

!!! note
	The installer will check the $MPICXX environment variable to determine a valid MPI C++ compiler. Also make sure you have the corresponding modules loaded in your system.

