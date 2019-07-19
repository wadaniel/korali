# Constrained Optimization

##Description

In a *Constrained Optimization* problem, the computational model is assumed to be of the form: $f:\mathbb{R}^N\rightarrow\mathbb{R}$, where $N$ is the number of variables **EXPLAIN THE CONSTRAINT ASPECT HERE**, as defined by the user. Given a set of values for the variables (a sample), the model produces a single real value containing the direct evaluation of $f(x)$. 


** Usage **

```python
  k["Problem"] = "Constrained Optimization"
```

##Requirements

+ A [Simple](/usage/models/simple) computational model should be defined.
+ At least one [constraint](/usage/models/simple) model should be defined. 
+ At least one variable should be defined.

##Settings

No other settings.