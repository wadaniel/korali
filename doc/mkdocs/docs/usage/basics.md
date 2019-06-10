# Introduction

##Programming with Korali

Korali provides scientists with a range of optimization and uncertainty quantification tools. To use these tools, a user needs to create a *Korali Application*. This application serves as a nexus between the computational models (user-provided C++ objects, Python modules, or pre-compiled applications), and the Korali's main solver engine, as illustrated below.

![](modules.png)

### Building a Korali Application with Python

To build a Korali application in Python, you simply need to include the corresponding korali module and declare an instance of the Korali engine:

```python
#!/usr/bin/env python3
import korali
k = korali.Engine()
...
```

### Building a Korali Application with C++

To build a Korali application in C++, you need to include the *korali.h* header file and instantiate an Korali::Engine object:

```c++
#include "korali.h"

int main(int argc, char* argv[])
{
 auto korali = Korali::Engine();
 ...
```

To compile and link, you should use the flags provided by the ```korali.cxx``` module:
```bash
> KORALICXX=`python3 -m korali.cxx --compiler`
> $KORALICXX -c myApp.cpp `python3 -m korali.cxx --cflags`
> $KORALICXX -o myApp myApp.o `python3 -m korali.cxx --libs`
```

## Running a Korali Application

A Python/C++ application can be run just like any other application, through a shell command:

```bash
> ./myApp myArguments
```

For distributed jobs, you can launch your Korali application with the SLURM/MPI/UPC++ launcher, specifying the number of cores, for example:

```bash
> srun -n 16 ./myApp myArguments
> mpirun -n 16 ./myApp myArguments
> upcxx-run -n 16 ./myApp myArguments
```

Once the Korali engine has started running, it will execute non-preemptively until a termination criterion is met. Each solver method exposes a different set of termination criteria which can be specified through Korali's [configuration](/usage/config). Some of these criteria are enabled by default, and others will only take effect if specified by the user.

## Processing Results

After execution, Korali will automatically create a results folder ```_korali\_result``` where it will save the entire state of the engine (i.e., a *checkpoint*) after each generation. Users can perform several operations on these results:

- [Export to Format](/usage/postprocessing/export)
- [Resume Execution](/usage/postprocessing/resume)
- [Generate Plots](/usage/postprocessing/plotting)