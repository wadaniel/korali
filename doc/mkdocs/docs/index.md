# Welcome to Korali

Korali is a high performance computing framework for optimization, sampling and Bayesian uncertainty quantification of large scale computational models.

<br><br>



# What Korali does for you



### [**1. Direct problems**](quick_start/direct.md)

Given any function $f:\mathbb{R}\rightarrow\mathbb{R}^N$ korali can

  - [**Optimize:**](quick_start/direct.md#optimize) find the minima or maxima of $f$
  $$
	\vartheta^\star = \mathop{\arg\min}\limits_{\vartheta}  \,\,\, f(\vartheta) \,,
	$$

  - [**Sample:**](quick_start/direct.md#sample) treat $f$ as an unnormalized density and draw samples from it
  $$
  \vartheta^{(k)} \sim f.
  $$  



### [**2. Bayesian Inference**](quick_start/bayesian.md)







2. [**Uncertainty Quantification**](quick_start/posterior.md): given a set of data $d$, respectively the output of a model $f(x;\vartheta)$, a likelihood function $p(d|\vartheta)$,  and a prior probability density $p(\vartheta)$ Korali samples the posterior distribution
	$$
	p(\vartheta | d) = \frac{p(d | \vartheta) \, p(\vartheta)}{p(d)}\, .
	$$
    I.e. the output is a set of samples $\theta^{(k)}$ with $k \in \{1,...,N\}$ such that
    $$
		\vartheta^{(k)} \sim p(\theta | d).
    $$
    Note that $x$ are the input parameters of the computational model $f$.

<br>

After [installing](installation.md) the software you can have a look at the [quick start](quick_start) guide or [examples](./examples/sampling.md) and learn how to run your own code.

<br><br>

# Korali Principles

(todo)

## Notes

Please send questions and feedback to:

- martiser AT ethz.ch
- garampat AT ethz.ch
- wadaniel AT ethz.ch


<br><br><br><br><br><br>

!!! warning
    The software and the documentation page are under continuous development. New pages and new feature will be constantly added.
