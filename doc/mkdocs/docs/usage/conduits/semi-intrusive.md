# Conduits / Semi-Intrusive

## Description

The semi-intrusive conduit exposes a simple interface for the execution of Python or C++ functions. This conduit requires users to develop an interface to the computational model that allows Korali to gather results directly from memory during runtime. 

The semi-intrusive conduit provides an efficient interface for passing samples and read results directly from the computational model's host memory without the intervention of the operating system for process creation or I/O operations. Furthermore, the simplicity of semi-intrusive conduit makes it ideal for accessible or open-source computational models for which a Korali-compatible interface can be easily developed.

## Usage

The semi-intrusive is the default option for a conduit in Korali. It executes a single instance of the computational model at a time, also returning its result to the solver one-by-one.

```python
#!/usr/bin/env python3
import korali
k = korali.Engine()

# Set problem, solver, variables, and model.
...

# Defining the semi-intrusive conduit.
# Not necessary since this is the default conduit
k["Conduit"] = "Semi-Intrusive"

k.run()
```

Then, we run our application normally:

```bash
> ./myKoraliApp
```
