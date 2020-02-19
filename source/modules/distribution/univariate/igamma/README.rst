
## Inverse Gamma

A random variable X is inverse gamma distributed with shape alpha and scale beta when
the 1/X follows gamma distributed with shape alpha and scale 1/beta.

Example:

```python
# Setting a Gamma prior distribution for variable 2
k["Variables"][2]["Bayesian"]["Prior Distribution"]["Type"] = "IGamma"
k["Variables"][2]["Bayesian"]["Prior Distribution"]["Shape"] = +2.5
k["Variables"][2]["Bayesian"]["Prior Distribution"]["Scale"] = +1.0
```
