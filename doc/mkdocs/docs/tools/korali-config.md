# configure

##Name
       korali-config - displays information about Korali's version and configuration.

##Usage
       ./korali-config
                   [--prefix]
				   [--python]
				   [--cxxlibs]			   
                   [--cxxflags]
				   [--cxxcompiler]
                   [--help]
				   [--version]

##Description

       Displays useful information about the current installation of Korali, including:

       --prefix  
 	       The current installation folder.

	   --python
           Displays information about the supported Python library and
		   required configuration to run a Korali application under Python.	   

	   --cxxlibs
  	       The libraries required to link a C++ Korali application.

       --cxxflags
	       The flags required to compile a C++ Korali application.

	   --cxxcompiler
	       The C++ compiler/wrapper required to build a C++ Korali Application.

       --help
   	       A help message displaying korali-config usage.

       --version
   	       The currently installed version number.     
