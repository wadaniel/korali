# configure

##Name
       korali-config - displays information about Korali's version and configuration.
	   
##Usage
       ./korali-config
                   [--prefix]
				   [--python]
				   [--clibs]			   
                   [--cflags]
				   [--cxx]
                   [--help]
				   [--version]
				   
##Description

       Displays useful information about the current installation of Korali, including:
           
       --prefix  
 	       The current installation folder.
		 
	   --python
           Displays information about the supported Python library and 
		   required configuration to run a Korali application under Python.	   

	   --libs
  	       The libraries required to link a C++ Korali application.
           
       --cflags 
	       The flags required to compile a C++ Korali application.

	   --cxx 
	       The C++ compiler/wrapper required to build a C++ Korali Application.

       --version 
   	       A help message displaying korali-config usage.
		   
       --version 
   	       The currently installed version number.     
