# Evaluation Type / Direct

##Description

In a Direct problem type, the computational model is assumed to be of the form: $f:\mathbb{R}^N\rightarrow\mathbb{R}$, where $N$ is the number of computational variables, as defined by the user. Given a set of values for the variables (a sample), the model produces a single real value containing the direct evaluation of $f(x)$. 


##Syntax
```python
  # Definition
  korali["Problem"]["Type"] = "Direct"
```	  

##Requirements

+ It should contain at least one computational variable as input for the computational model.
+ It should contain no statistical variables.
+ The computational model should return exactly 1 result. 

##Settings

No other settings.