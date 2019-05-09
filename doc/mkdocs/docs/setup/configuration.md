# Configuration

This section details the options for the configure command provided to build Korali.

##Name
       configure - configuration script to compile and install Korali.

##Usage
       ./configure
                   [--prefix=INSTALL_DIR]
                   [--enable-upcxx]
                   [--enable-mpi]
                   [--use-gsl=PATH_TO_GSL]
                   [--help]

##Description

       configure sets the installation environment required to build
       and install Korali. It performs the following tasks:

       1   Determines whether a compatible C++ compiler exists.
	       Korali will check for the $CXX environment variable to determine the
		   default C++ compiler and wheter it supports POSIX Threads.
		   Modify $CXX to use a custom C++ compiler for Korali.

       2   Determines whether a compatible Python3 interpreter exists.
	       Korali will use the 'which python3' command to determine location, version,
		   libraries, and compilation flags to Python. Make sure this command is
		   reachable from the $PATH environment variable to enable support for python.

       3   [Optional] Determine whether the UPC++ library is installed. Korali will
	       use the 'upcxx' command to find UPC++ and, if found, enable the UPC++
		   conduit for distributed sampling.

	   4   [Optional] Determine whether an MPI library is installed. Korali will
	       use the 'upcxx' command to find MPI and, if found, enable the MPI conduit for
		   distributed sampling.

	   5   [Optional] Determine whether the GSL++ library provided by path is installed.

##Options

       --prefix=INSTALL_PATH
              Specifies where Korali and its dependencies will be installed.
              Default value: $PWD/Korali

       --enable-upcxx
              Determines whether Korali will install with the UPC++ conduit enabled.
              Default value: Disabled

	   --enable-mpi
              Determines whether Korali will install with the MPI conduit enabled.
              Default value: Disabled

	   --use-gsl=PATH_TO_GSL
              Disables automatic download and installation of GSL, and instead uses the
			  provided GSL library.
              Default value: Disabled

       --help
              Displays a help message.

##Make Commands

       After the configuration script was been executed without errors, the user can compile
       and install Korali using the following make commands:

       make
           Compiles Korali and its dependencies. It downloads and compiles
           Korali's dependencies. If dependencies are already installed, it does not re-build them.

       make install
           Installs Korali in the directory specified with the --prefix option.

       make test
           Compiles and executes all tests included in the tests/ folder. It is recommended
		   that this command is executed after installing Korali to verify that it runs correctly.

       make clean
           Cleans Korali's compilation files, but does not uninstall it from the
		   installation directory.

##Dependencies
       All Korali's dependencies are automatically downloaded and compiled by the
	   installation script.

       Some dependencies require that some GNU programs are already installed
       in the system. Korali's configuration script does not check for all of these
       sub-dependencies, but an error message informing which programs are missing
       will be displayed by the dependency's own configure script.  

       Python 3.
          Korali requires an existing installation of Python (version > 3.0) accessible
		  through the 'python3' command. Korali will look for its associated pkg-config
		  file (python3.pc) to determine CFLAGS and LIBS for compilation with python support.
		  Korali's plotting tools also require a working installation of python3 to work.

       GNU Scientific Library (GSL).
          The GNU Scientific Library is a software library for numerical computations
		  in applied mathematics and science. If Korali does not find a compatible GSL
		  installation currently installed in the system, it will download it and compile
		  it during installation.

	   (NLohmann) Json for C++
		  A library for management of Json structured in C++ code. The main header file,
          obtained from https://github.com/nlohmann/json, is already included in Korali's source
