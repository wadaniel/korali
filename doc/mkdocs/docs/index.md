# Korali  ![](images/templogo.png){: style="height:25px;width:25px"}

**Korali** is a high-performance computing framework for optimization, sampling and Bayesian uncertainty quantification of large-scale computational models.

## What Korali does for you

### [Optimization](/tutorials/optimization/optimization)
Given any function $f:\mathbb{R}^N\rightarrow\mathbb{R}$ Korali can find its minima or maxima: $\vartheta^\star = \mathop{\arg\min}\limits_{\vartheta}  \,\,\, f(\vartheta) \,,$

### [Sampling](/tutorials/sampling/sampling)
Treat $f$ as an unnormalized density and draw samples to determine its distribution: $\vartheta^{(k)} \sim f.$  

### [Bayesian Inference](/tutorials/bayesian/bayesian)

Given a set of data $d$, the output of a computational model $f(x;\vartheta)$, a likelihood function $p(d|\vartheta)$,  and a prior probability density $p(\vartheta)$ Korali will build the posterior distribution:
$$
p(\vartheta | d) \, = \, \frac{p(d | \vartheta) \,\, p(\vartheta)}{p(d)}\,
$$

Allowing users to compute the maximum a posteriori estimate: $\vartheta^\star = \mathop{\arg\min}\limits_{\vartheta}  \,\,\, p(\theta | d) \,,$ or sample its posterior distribution: $\vartheta^{(k)} \sim p(\theta | d).$
