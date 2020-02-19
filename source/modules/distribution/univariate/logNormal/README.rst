*******************************
LogNormal Distribution
*******************************

## Lognormal

The probability density of the Lognormal is

$${\displaystyle f(x\mid \mu ,\sigma ^{2}) = \frac 1 {x\sigma\sqrt{2\pi}}\ \exp\left(-\frac{\left(\ln x-\mu\right)^2}{2\sigma^2}\right)  }$$

Where:

$\mu\in(-\infty,\infty)$  and
$\sigma > 0$.

Example:

```python
# Setting a Normal prior distribution for variable 3
k["Variables"][3]["Bayesian"]["Prior Distribution"]["Type"] = "Lognormal"
k["Variables"][3]["Bayesian"]["Prior Distribution"]["Mu"] = -1.0
k["Variables"][3]["Bayesian"]["Prior Distribution"]["Sigma"] = +2.0
```
