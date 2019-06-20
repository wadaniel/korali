# Distributions

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

## Exponential
   				   
The Exponential distribution has the probability density function (PDF):

$$f(x;\beta) = \begin{cases}
\frac{1}{\beta} e^{-\frac{x}{\beta}} & x \ge 0, \\
0 & x < 0.
\end{cases}$$

Where $\beta > 0$ is mean, standard deviation, and scale parameter of the distribution, the reciprocal of the rate parameter, $\lambda$, defined above. In this specification, $\beta$ is a survival parameter in the sense that if a random variable X is the duration of time that a given biological or mechanical system manages to survive and $X ~ Exp(\beta)$ then $E[X] = \beta$.

Example:

```python
# Setting an Exponential prior distribution for variable 1
k["Variables"][1]["Bayesian"]["Prior Distribution"]["Type"] = "Exponential"
k["Variables"][1]["Bayesian"]["Prior Distribution"]["Mean"] = +3.0
k["Variables"][1]["Bayesian"]["Prior Distribution"]["Location"] = -1.8
```

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

## Laplace
   				   
The probability density of the Laplace distribution is

$$f(x\mid \mu ,b)={\frac  {1}{2b}}\exp \left(-{\frac  {|x-\mu |}{b}}\right)\,\!$$
$$={\frac  {1}{2b}}\left\{{\begin{matrix}\exp \left(-{\frac  {\mu -x}{b}}\right)&{\text{if }}x<\mu \\[8pt]\exp \left(-{\frac  {x-\mu }{b}}\right)&{\text{if }}x\geq \mu \end{matrix}}\right.$$

Where:

$\mu$ is a the mean parameter and $b > 0$ is a width parameter. If $\mu=0$ and $b=1$, the positive half-line is exactly an exponential distribution scaled by 1/2.

Example:

```python
# Setting a Laplace prior distribution for variable 4
k["Variables"][4]["Bayesian"]["Prior Distribution"]["Type"] = "Laplace"
k["Variables"][4]["Bayesian"]["Prior Distribution"]["Mean"] = +1.5
k["Variables"][4]["Bayesian"]["Prior Distribution"]["Width"] = +2.0
```

## Uniform
   				   
The probability density function of the continuous uniform distribution is:

$$f(x)={\begin{cases}{\frac {1}{b-a}}&\mathrm {for} \ a\leq x\leq b,\\[8pt]0&\mathrm {for} \ x<a\ \mathrm {or} \ x>b\end{cases}}$$

Where:

$a$ represents the minimum (lower bound), and $b$ represents the maximum (upper bound) of the distribution.


Example:

```python
# Setting a Uniform prior distribution for variable 5
k["Variables"][5]["Bayesian"]["Prior Distribution"]["Uniform"] = "Uniform"
k["Variables"][5]["Bayesian"]["Prior Distribution"]["Minimum"] = -8.0
k["Variables"][5]["Bayesian"]["Prior Distribution"]["Maximum"] = +16.0
```