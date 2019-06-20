# Conduits / Distributed

The distributed conduit provides support for MPI distributed computational models. It works by defining a set of evaluation teams (see Figure). Evaluation teams comprise multiple cores (i.e.,MPI or UPC++ ranks) each which receivenew samples to be evaluated and return their results asynchronously. The conduit will distribute newsamples to teams as soon as they become free. By distributing sample arguments to cores speculatively, this conduit reduces system-wide load imbalance and communication overhead.

![](distributedConduit.png)

## Distributed Execution of Sequential Models

In this case, we want to execute *N* parallel instances of a sequential model, with only 1 rank per computational model evaluation. We define a simple Korali application:

```python
#!/usr/bin/env python3
import korali
k = korali.Engine()

# Set problem, solver, variables, and model.
...

# Defining the distributed conduit
k["Conduit"] = "Distributed"

# Setting ranks per team to 1.
# This is not really necessary since the default value is 1.
k["Distributed"]["Ranks Per Team"] = 1

k.run()
```

And then run the application in shell, using the corresponding job launcher. We need N+1 ranks since the Korali engine requires a rank to serve as master:

```bash
> mpirun -n N+1 ./myKoraliApp
```

## Single Instance of a Distributed Model

In this case, we want to execute a single instance of a sequential model at a time, with *M* ranks per computational model evaluation. We define another Korali application:

```python
#!/usr/bin/env python3
import korali
k = korali.Engine()

# Set problem, solver, variables.
...

#Using a distributed (MPI) model
def myModel(data): 
 # Grabbing a pointer to the team's MPI communicator
 x = data.getVariable(0)
 y = data.getVariable(1)
 comm = korali_obj.getCommPointer()
 fval = myMPIModel(comm, x0, ks)
 data.addResult(fval)
 
k.setModel(myModel);

# Defining the distributed conduit
k["Conduit"] = "Distributed"

# Setting ranks per team to M.
k["Distributed"]["Ranks Per Team"] = M

k.run()
```

And then run the application in shell, using the corresponding job launcher:

```bash
> mpirun -n M+1 ./myKoraliApp
```

## Parallel Execution of a Distributed Model

In this case, we want to execute N instance of a sequential model at a time, with *M* ranks per computational model evaluation. We define another Korali application:

```python
#!/usr/bin/env python3
import korali
k = korali.Engine()

# Set problem, solver, variables.
...

#Using the same MPI model as above.
k.setModel(myModel);

# Defining the distributed conduit
k["Conduit"] = "Distributed"

# Setting ranks per team to M.
k["Distributed"]["Ranks Per Team"] = M

k.run()
```

And then run the application in shell, using the corresponding job launcher:

```bash
> mpirun -n N*M+1 ./myKoraliApp
```
