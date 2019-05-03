# Welcome to Korali

Korali is a high performance computing framework for optimization, sampling and Bayesian uncertainty quantification of large scale computational models.

Korali is based on [UPCXX](https://bitbucket.org/berkeleylab/upcxx/wiki/Home)

<br><br>

# What Korali does for you

1. [**Gradient-Free Black-Box Optimization**](quick_start/direct.md): given a cost function $F(\vartheta)$ find
	$$
	\vartheta^\star = \mathop{\arg\min}\limits_{\vartheta}  \,\,\, F(\vartheta) \,,
	$$
    where $F(\theta)$ can be any nonlinear nonconvex function.


2. [**Uncertainty Quantification**](quick_start/posterior.md): given a set of data $d$, respectively the output of a model $f(x;\vartheta)$, a likelihood function $p(d|\vartheta)$,  and a prior probability density $p(\vartheta)$ Korali samples the posterior distribution
	$$
	p(\vartheta | d) = \frac{p(d | \vartheta) \, p(\vartheta)}{p(d)}\, .
	$$
    I.e. the output is a set of samples $\theta^{(k)}$ with $k \in \{1,...,N\}$ such that
    $$
		\vartheta^{(k)} \sim p(\theta | d).
    $$
    Note that $x$ are the input parameters of the computational model $f$.

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
