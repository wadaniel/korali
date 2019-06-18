We have successfully tested Korali in the following environments:

  - g++ (GCC) 6.3.0, CentOS Linux 7
  - g++ (GCC) 8.2.0, CentOS Linux 7
  - Apple LLVM version 10.0.1 (clang-1001.0.46.4), macOS 10.14.4 (Darwin 18.5.0)
  - Cray Programming Environment 2.5, SUSE Linux Enterprise Server 12 SP3
  
  You can verify your C++ compiler and OS versions with the following commands:

```shell
# For Linux
g++ --version
cat /etc/os-release

# For MacOS
clang++ --version
system_profiler SPSoftwareDataType
```

We have successfully tested Korali with the following Python versions:

  - Python 3.3.0
  - Python 3.7.0
  - Python 3.7.1