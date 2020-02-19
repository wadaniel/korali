*******************************
Laplace Distribution
*******************************

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




