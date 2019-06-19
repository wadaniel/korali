# Computational Model

During execution, the Korali engine will evaluate many *samples*. A Korali sample is a vector containing a particular set of values for all (computational and statistical) variables defined in the problem. Korali will stochastically produce new samples to be evaluated by a user-provided computational model.


## Interface

Korali exposes a single interface for computational models, regardless of the chosen conduit/evaluation type. The computational model is specified via Korali's ```setModel(myModel)``` function, which receieves a function pointer that accepts a single argument of type ```KoraliData```.

Example for C++:

```C++
// Using a traditional C++ function pointer
void myModel(KoraliData& data) { /* do stuff */ };
k.setModel(&myModel);

// Using a lambda expression
auto myModel = [](Korali::ModelData& data) { /* do stuff */ };
k.setModel(myModel);
```

Example for Python:
```python
# Using a normal function definition
def myModel(data): #do stuff
k.setModel(myModel);

# Using a lambda expression
myModel = lambda koraliData: /* do stuff */ 
k.setModel(myModel);
```

## Reading Parameters 
The ```KoraliData``` object exposes information about the sample's computational variables. The snippet below shows the relevant operations:

```python
data.getVariableCount() # Return the number of computational variables in the sample
data.getVariables() # Returns a vector containing all the computational variables
data.getVariable(i) # Returns the value of the ith computational variable
data.getHashId() # Returns a unique identifier for the sample
```

In the code below shows how a computational model can access sample data:

```python
# This computational model sums the squares of all sample arguments.
def myModel(data):
 sum = 0.0;
 for i in range(data.getVariableCount()):
  sum += x.getVariable(i)*x.getVariable(i)
```

## Saving Results

Korali's engine requires the computational model to produce at least one result to be sent back to the Engine (depending on the evaluation type, it may require more than one result). To save the results of the evaluation, Korali exposes the following interface:

```python
data.addResult() # Adds a new result to the vector result to be sent back to Korali's engine
data.getResults() # Returns a the full results vector by reference
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

In the code below shows how a computational model saves a multiple results:

```python
# This computational model returns the squares for all sample arguments.
def myModel(data):
 for i in range(data.getVariableCount()):
  data.addResult(x.getVariable(i)*x.getVariable(i))
```