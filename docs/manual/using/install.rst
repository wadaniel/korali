*********************
Installation
*********************

Installation Steps
====================

  1. Download Korali

    Download the latest stable release of Korali with the following command:
   
     .. code-block:: bash
    
         git clone https://github.com/cselab/korali.git --branch latest

  2. Install Korali

    To build and install Korali, simply run:

      .. code-block:: bash

        cd korali
        ./install

Troubleshooting
====================

  If you are experiencing problems installing or running Korali, please check the following hints:

  - Check Korali's `system requirements <#system-requirements>`_ to verify that your system has all the required software components.

  - Check the `build status </korali/docs/dev/testing.html>`_  to see if Korali is currently building correctly.

  - If the problem persists, please submit a new `issue report <https://github.com/cselab/korali/issues>`_ on our Github repository detailing the problem, your system information, and the steps to replicate it and we will promptly address it.

  - For further questions, feel free to `contact us </korali/#contact>`_.

System Requirements
====================

Mandatory Requirements
---------------------------

  - **C++ Compiler**
      Korali requires a C++ compiler that supports the C++14 standard (`-std=c++14`) to build.
      **Hint:** Check the following `link <https://en.cppreference.com/w/cpp/compiler_support#C.2B.2B14_core_language_features>`_ to verify whether your compiler supports C++14.
      Korali's installer will check the **$CXX** environment variable to determine the default C++ compiler. You can change the value of this variable to define a custom C++ compiler.
  
  - **GNU Make**
      Korali requires GNU Make to build its C++ engine.
  
  - **wget**
      Korali requires access to the internet and the *wget* command to be available to automatically resolve some of its dependencies. If you wish to install Korali on a system without access to internet or no *wget* command, you check the `Installed by Korali <#automatically-installed-by-korali>`_ section of this document to manually define these requirements.
  
  - **Git Client**
      You need Git to clone (download) our code before installation.
  
  - **Python3**
      Korali requires a version of Python higher than 3.0 to be installed in the system. Korali's installer will check the *python3* command. The path to this command should be present in the $PATH environment variable. *Hint:* Make sure Python3 is correctly installed or its module loaded before configuring Korali.
  
  - **python3-config**
      Korali requires the command *python3-config* to be available during installation. This command is typically included in standard installations of python3 that include developer tools. *Hint:*  If *python3-config* is missing, you can get it by installing/loading the **python3-dev** package/module in your system.
  
  - **Pip3 Installer**
      Korali requires the *pip3* command to install it's engine and tools. This command is typically included in standard installations of python. *Hint:*  If *pip3* is missing, you can get it by installing pip3, with e.g, ``brew install pip3``
  
Automatically Installed by Korali
------------------------------------
  
  - **PyBind11**
      Korali requires *pybind11* to enable Python/C++ interaction. If not found, it will try to install it automatically using *pip3*.
  
  - **GNU Scientific Library**
      Korali requires the `GSL-2.4 <http://www.gnu.org/software/gsl/>`_ or later must be installed on your system. If the command ``gsl-config`` is not found, Korali will try to install GSL automatically. *Hint:* If you do not want Korali downloading and installing GSL on its own, run ``./install --no-prereqs``

Optional Requirements
---------------------------------

  - **MPI Library**
      One way to enable support distributed conduits and computational models is to configure Korali to compile with an MPI compiler. The installer will check the *$MPICXX* environment variable to determine a valid MPI C++ compiler.

