# configure

##Name
       configure - configuration script to compile and install Korali.
	   
##Usage
       ./configure --with-mpi=MPI_PATH
                   [--prefix=INSTALL_DIR]
                   [--build-dir=BUILD_DIR]
                   [--force]
                   [--help]
##Description
       configure sets the installation environment required to build
       and install Korali. It performs the following tasks:
       
       1   Define the path where Korali and its dependencies (ROSE, Boost,
           Tarragon) are going to be built. This path needs to have 
           with at least 5Gb of free space prior to compilation.
           
       2   Verify that the MPI library binary and include paths exist.
           
       3   Determine which version of Boost to use depending on the current gcc
           version.
           
       4   Verify that some of the tool dependencies used by Korali or ROSE can be
           executed.
      
##Options
       These options are used to choose the information displayed by ps.
       The output may differ by personality.

       --with-mpi=MPI_PATH
              Specifies where the MPI library to use is installed. Configure will look
              for and execute MPI_PATH/bin/mpicxx command, and verify that the file 
              MPI_PATH/include/mpi.h exists. This MPI library will be used by Korali
              to compile and link the translated source files.
    
       --prefix=INSTALL_PATH
              Specifies where Korali and its dependencies will be installed. INSTALL_PATH
              needs to have at least 1Gb of free space at the moment of installation.
              Default value: $PWD/bin

       --build-dir=BUILD_PATH
              Specifies where Korali and its dependencies will be downloaded and built.
              BUILD_PATH needs to have at least 1Gb of free space at the moment
              of installation.
              Default value: $PWD/build

       --help
              Displays a help message.
##EXAMPLES
       Simple configuration example:
          ./configure --with-mpi=/opt/mpich2

       Simple configuration example with recommended simultaneous job number:
          ./configure --with-mpi=/opt/mpich2 --build-jobs=16

       Configuration with different filesystems for build/installation:
          ./configure --with-mpi=/opt/mpich2 --build-jobs=16 --build-dir=/big-filesystem/Korali-build --prefix=/usr/bin/Korali

       Display help message:
          ./configure --help
		  
##MAKE COMMANDS

       After the configuration script was been executed without errors, the user can compile
       and install Korali using the following make commands:
       
       make
           Compiles Korali and its dependencies. Creates the --build-dir path to download and compile
           dependencies. If dependencies are already installed, it does not re-build them.
       
       make install
           Installs Korali in the directory specified with the --prefix option.

       make check
           Compiles and executes all tests included in the tests/ folder. It is recommended that this
           command is executed after installing Korali to verify that it runs correctly.
           
       make clean
           Cleans Korali's compilation files, but does not uninstall it from the installation directory.
		   
##COMPATIBILITY
       The following combinations of gcc/boost have been tested, and the correct
       combination is chosen from the gcc version being used. Automatic fixes to ROSE
       installation are applied in some cases:

         gcc 4.4.*: boost 1.50
         gcc 4.5.*: boost 1.47 + fixEDGbuild.sh (*)
         gcc 4.6.0: boost 1.47 + fixEDGbuild.sh (*)
         gcc 4.6.*: boost 1.53 + fixEDGbuild.sh (*)
         gcc 4.7.*: boost 1.53
         gcc 4.8.*: boost 1.53
         gcc 4.9.*: boost 1.53 + fixEDGbuild.sh + fixgcc49x.sh (*) (**)
         
       Using gcc for compilation of Korali and its dependencies is mandatory since 
       the ROSE compiler infrastructure can only be compiled with gcc. 

       There are no restrictions on which MPI library can be used for Korali, although
       the majority of tests have been performed using MPICH. 
	   
       (*)  The script fixEDGbuild.sh is automatically used to select the correct
            version of EDG binaries compatible with the current gcc version used
            to compile ROSE.
       (**) The script fixgcc49x.sh is automatically used to allow ROSE use gcc 4.9.*
	   
##DEPENDENCIES
       All Korali's dependencies are automatically downloaded and compiled by the
	   installation script. 
       
       Some dependencies require that some GNU programs are already installed
       in the system. Korali's configuration script does not check for all of these
       sub-dependencies, but an error message informing which programs are missing
       will be displayed by the dependency's own configure script.  

       GNU Scientific Library (GSL).
          The GNU Scientific Library is a software library for numerical computations
		  in applied mathematics and science.