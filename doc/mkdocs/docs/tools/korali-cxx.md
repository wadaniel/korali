# korali-cxx

##Name
       korali-cxx - Compiler/linker wrapper for C++ Korali applications. 
	   
##Usage
       korali-cxx [cxxArgs]

				   
##Description

       korali-cxx is an alias for the underlying compiler CXX command specified during
       Korali's configuration. It adds additional flags, includes, and linked libraries
       that are necessary for C++ Korali applicationse to compile, and to link with
       the Korali's engine and conduits. 
       
       When korali-cxx is executed, all its arguments are redirected to CXX with appended
	   Korali-specific arguments. If the UPC++ conduit is employed, Korali will use
	   upcxx as underlying compiler.

##Options

      These options are used to specify source files and mpicxx flags to korali-cxx.

      cxxArgs
        Specifies the original CXX/Linker flags required to compile the Korali application.
	
##Examples
       Single file compilation and linking:
          korali-cxx myapp.cpp -o myapp

       Multiple file compilation and linking:
          korali-cxx myapp.cpp aux1.c aux2.c -o myapp

       Multiple file compilation and linking in separate stages:
          korali-cxx -c myapp.cpp
          korali-cxx -c aux1.c
          korali-cxx -c aux2.c
          korali-cxx myapp.o aux1.o aux2.o -o myapp

       Multiple file compilation and linking in separate stages
       with include directories, definitions, additional libraries, and flags:
          korali-cxx -c myapp.cpp -Dvalue=100 -O2
          korali-cxx -c aux1.c -I/usr/include -O2
          korali-cxx -c aux2.c -O2
          korali-cxx myapp.o aux1.o aux2.o -o myapp -L/user/lib -lm
		  
##Execution
       If the Korali application is configured to use the Sequential conduit, it can
	   can be run just like any other application:
	   
       ./myapp [myArguments]
	   
	   If the Korali application is configured to use the Multithreaded conduit, the user
	   can define the number of threads through the KORALI_THREADS environment variable:
	   
       export KORALI_THREADS=16
	   ./myapp [myArguments]
	   
	   If the Korali application is configured to use the UPC++ conduit, it should be instantiated
	   using UPC++'s run command. For example:
	   
	   upcxx-run -n NRANKS ./myApp [myArguments]
	   
	   Similarly, if the application is configured to use the MPI conduit, it should be instantiated
	   using MPI's (or job batch) run command. For example:
	   
	   mpirun -n NRANKS ./myApp [myArguments]
	   srun -n NRANKS ./myApp [myArguments]
	   ibrun -n NRANKS ./myApp [myArguments]
	   
	   