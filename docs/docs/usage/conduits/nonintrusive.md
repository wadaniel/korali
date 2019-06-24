# Conduits / Semi-Intrusive

## Description

Many computational models are comprised of legacy codes that can be too complex or poorly-documented for non-expert users to modify. Thus, extending them with a Korali-compatible interface may be unfeasible. Other commercial / pre-compiled applications may not even allow users to access their code. For these models, we provide a non-intrusive conduit that allows running pre-compiled models via shell commands and reading back the results through either file or pipe I/O operations. This conduit can be used to launch and gather results from large-scale parallel (MPI/UPC++) applications.

Although Korali's non-intrusive conduit provides flexibility in the execution of any computational model, it does require that the user specifies how to pass the sample's values as inputs to the model. Furthermore, it requires the user to specify how to parse the results from file or standard output. Korali's non-intrusive conduit uses a fork/join strategy to instantiate an independent operating-system process per each sample. Since I/O and process management operations incur much more overhead than direct function calls, the non-intrusive conduit may be less efficient than the semi-intrusive ones, especially when launching a large number of short samples.

## Usage

### Launching an External Application.

The Nonintrusive conduit guarantees that each model is executed as an independent process. As such, the model can be suspended in wait for a sub-process to run, as shown in the following example:

```python
def myModel(x):
  argString = ['./model/ackleyFunction.py']
  for i in range(x.getVariableCount()):
    argString.append(str(x.getVariable(i)))
  retValue = subprocess.check_output(argString)
  x.addResult(float(retValue.decode()))
```

The model in the example launches a new process that solves the Ackley function and outputs the result to Stdout. The model captures the output and returns it to Korali. Note that, although the model is suspended until the subprocess finishes, Korali continues evaluating samples normally.

The Korali application is then programmed as usual:

```python
#!/usr/bin/env python3
import korali
k = korali.Engine()

# Set problem, solver, and variables.
...

# Set model.
k.setModel(myModel)

# Defining the Nonintrusive conduit
k["Conduit"] = "Nonintrusive"

# Setting concurrent jobs count. 
# If this is larger than 1, then many samples will be evaluated in parallel
# as separate processes. In this case, execution is limited to a single node.
k["Conduit"]["Concurrent Jobs"] = 16

k.run()
```

Then, we run our application normally:

```bash
> ./myKoraliApp # Will run 16 processes!
```

### Launching an MPI Application.

In the case of MPI processes, the procedure is similar, but Korali needs to be launched within the scope of a job (e.g., SLURM). An example model is shown below:

```python
def myModel(x):
  argString = ['mpirun -n 16 ./model/ackleyMPIFunction.py']
  for i in range(x.getVariableCount()):
    argString.append(str(x.getVariable(i)))
  retValue = subprocess.check_output(argString)
  x.addResult(float(retValue.decode()))
```

