*******************************
Cauchy Distribution
*******************************

## Cauchy

The Cauchy distribution has the probability density function (PDF):

$$f(x;x_{0},\gamma )={1 \over \pi \gamma }\left[{\gamma ^{2} \over (x-x_{0})^{2}+\gamma ^{2}}\right]$$

where $x_{0}$ is the location parameter, specifying the **location** of the peak of the distribution, and $\gamma$ is the **scale** parameter which specifies the half-width at half-maximum (HWHM).

Example:

```python
# Setting a Cauchy prior distribution for variable 0
k["Variables"][0]["Bayesian"]["Prior Distribution"]["Type"] = "Cauchy"
k["Variables"][0]["Bayesian"]["Prior Distribution"]["Location"] = +8.0
k["Variables"][0]["Bayesian"]["Prior Distribution"]["Scale"] = +2.0
```
