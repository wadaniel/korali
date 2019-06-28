# System Requirements

## Mandatory Requirements

??? abstract "C++ Compiler"

	Korali requires a C++ that supports the C++14 standard (`-std=c++14`) to build.

	**Hint:** Check the following [resource](https://en.cppreference.com/w/cpp/compiler_support#C.2B.2B14_core_language_features) to verify whether your compiler supports C++14.

	Korali's installer will check the **$CXX** environment variable to determine the default C++ compiler. You can change the value of this variable to define a custom C++ compiler.

??? abstract "GNU Make"

	Korali requires GNU Make to build its C++ engine.
	
??? abstract "wget"

	Korali requires access to the internet and the **wget** command to be available to automatically resolve some of its dependencies. 
	
	If you wish to install Korali on a system without access to internet or no **wget** command, you check the 'Installed by Korali' section of this document to manually define these requirements.
	
??? abstract "Git Client"

	You need Git to clone (download) our code before installation.
	
??? abstract "Python3"

	Korali requires a version of Python higher than 3.0 to be installed in the system.

	Korali's installer will check the **python3** command. The path to this command should be present in the $PATH environment variable. 
	
	**Hint:** Make sure Python3 is correctly installed or its module loaded before configuring Korali.

??? abstract "python3-config"
	
	Korali requires the command **python3-config** to be available during installation.
	
	This command is typically included in standard installations of python3 that include developer tools.
	
	**Hint:**  If **python3-config** is missing, you can get it by installing/loading the **python3-dev** package/module in your system.

??? abstract "Pip3 Installer"	
	
	Korali requires the **pip3** command to install it's engine and tools. 
	
	This command is typically included in standard installations of python.
	
	**Hint:**  If **pip3** is missing, you can get it by installing pip3, with e.g, 'brew install pip3'


### Installed by Korali	

??? abstract "PyBind11"

	Korali requires **pybind11** to enable Python/C++ interaction.

	If not found, it will try to install it automatically using **pip3**.

??? abstract "GNU Scientific Library"

	Korali requires the [GSL-2.4](http://www.gnu.org/software/gsl/) or later must be installed on your system. 
	
	If the command ```gsl-config``` is not found, Korali will try to install GSL automatically.
	
	**Hint:** If you do not want Korali downloading and installing GSL on its own, run './install --no-prereqs'
	
## Optional Requirements
	
??? abstract "MPI Library"

	One way to enable support distributed conduits and computational models is to configure Korali to compile with an MPI compiler.

	The installer will check the **$MPICXX** environment variable to determine a valid MPI C++ compiler. 

??? abstract "UPC++ Library"

	One way to enable support distributed conduits and computational models is to configure Korali to compile with an UPC++ compiler.

	The installer will check the **upcxx-meta** command avaiable through $PATH to determine a valid UPC++ compiler. 

