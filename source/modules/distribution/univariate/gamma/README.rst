*******************************
Gamma Distribution
*******************************

## Gamma
          
A random variable X that is gamma-distributed with shape k and scale θ is denoted by

$${\displaystyle X\sim \Gamma (k,\theta )\equiv \operatorname {Gamma} (k,\theta )}$$

The Gamma distribution has the probability density function (PDF) using a shape-scale parametrization is:

$$f(x;k,\theta) =  \frac{x^{k-1}e^{-\frac{x}{\theta}}}{\theta^k\Gamma(k)} \quad \text{ for } x > 0 \text{ and } k, \theta > 0.$$

Example:

```python
# Setting a Gamma prior distribution for variable 2
k["Variables"][2]["Bayesian"]["Prior Distribution"]["Type"] = "Gamma"
k["Variables"][2]["Bayesian"]["Prior Distribution"]["Shape"] = +2.5
k["Variables"][2]["Bayesian"]["Prior Distribution"]["Scale"] = +1.0
```



