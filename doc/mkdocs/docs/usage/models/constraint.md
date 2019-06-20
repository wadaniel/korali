# Models / Constraint

A Constraint is function of the form: $f:\mathbb{R}^N\rightarrow\mathbb{R}$, where *N* inputs, representing the variables values of a sample, that returns:

$$f(x)= \left\{{\begin{matrix} >=0 \quad \quad  \quad  \quad {\text{if x satisfies the given constraint.}} \\ < 0 \quad \text{if x does not satisfy the given constraint.} \end{matrix}}\right.$$


###Defining the Model

```pytho
def myConstraint(x, N): 
 #do stuff
 
k.addConstraint(myConstraint);
```

###Reading Parameters 

The constraint function receives an array of double precision floating points and the number of points *N*

In the code below shows how a constraint model can access sample data:

```python
# This computational model sums the squares of all sample arguments.
def myConstraint(x, N):
 sum = 0.0;
 for i in range(N):
  sum += x(i)
```

###Returning Result

The reference model interface requires a single constraint result to be returned. In the code below shows how an example:

```python
# This computational model returns the squares for all sample arguments.
# This computational model sums the squares of all sample arguments.
def myConstraint(x, N):
 sum = 0.0;
 for i in range(N):
  sum += x(i)
 return sum
```