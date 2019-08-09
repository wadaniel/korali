
## Gaussian
          
The probability density of the Gaussian distribution is

$${\displaystyle f(x\mid \mu ,\sigma ^{2})={\frac {1}{\sqrt {2\pi \sigma ^{2}}}}e^{-{\frac {(x-\mu )^{2}}{2\sigma ^{2}}}}}$$

Where:

$\mu$  is the mean or expectation of the distribution (and also its median and mode),
$\sigma$  is the standard deviation, and
$\sigma ^{2}$ is the variance.

Example:

```python
# Setting a Gaussian prior distribution for variable 3
k["Variables"][3]["Bayesian"]["Prior Distribution"]["Type"] = "Gaussian"
k["Variables"][3]["Bayesian"]["Prior Distribution"]["Mean"] = -3.5
k["Variables"][3]["Bayesian"]["Prior Distribution"]["Sigma"] = +1.0
```




