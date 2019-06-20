# Conduits / Distributed

The distributed semi-intrusive provides support for MPI, CUDA, UPC++, and OpenMP parallel computational models. It works by defining a set of evaluation teams (see Figure). Evaluation teams comprise multiple cores (i.e.,MPI or UPC++ ranks) each which receivenew samples to be evaluated and return their results asynchronously. The conduit will distribute newsamples to teams as soon as they become free. By distributing sample arguments to cores speculatively, this conduit reduces system-wide load imbalance and communication overhead.

![](distributedConduit.png)

## Settings

```python
  # Definition
  k["Conduit"] = "Distributed";
  
  # Solver Settings
  k["Distributed"]["Objective"] = ... 
  k["Distributed"]["Sample Count"] = ...
```

###Defining the Model

```python
# Using a normal function definition
def myModel(data): 
 #do stuff
 
k.setModel(myModel);

# Using a lambda expression
myModel = lambda koraliData: 
 #do stuff
 
k.setModel(myModel);
```

###Reading Parameters 

The ```KoraliData``` object exposes information about the sample's computational variables. The snippet below shows the relevant operations:

```python
data.getVariableCount() # Return the number of computational variables in the sample
data.getVariables() # Returns a vector containing all the computational variables
data.getVariable(i) # Returns the value of the ith computational variable
data.getHashId() # Returns a unique identifier for the sample
``

In the code below shows how a computational model can access sample data:

```python
# This computational model sums the squares of all sample arguments.
def myModel(data):
 sum = 0.0;
 for i in range(data.getVariableCount()):
  sum += x.getVariable(i)*x.getVariable(i)
```

###Returning Results

The simple model interface requires a single result to be sent back to the solver. To save the result of the evaluation, Korali exposes the following interface:

```python
data.setResult() # Sets the result of the computation
```

In the code below shows how a computational model saves a single result:

```python
# This computational model sums the squares of all sample arguments.
def myModel(data):
 sum = 0.0;
 for i in range(data.getVariableCount()):
  sum += x.getVariable(i)*x.getVariable(i)
 data.addResult(sum)
```