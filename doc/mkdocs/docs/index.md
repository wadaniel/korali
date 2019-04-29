# Welcome to Korali

Korali is a high performance computing framework for optimization, sampling and Bayesian uncertainty quantification of large scale computational models.

Korali is based on [UPCXX](https://bitbucket.org/berkeleylab/upcxx/wiki/Home)

<br><br>


# What Korali does for you

1. **Gradient-Free Black-Box Optimization**: given a cost function $F(\vartheta)$ find
	$$
	\vartheta^\star = \mathop{\arg\min}\limits_{\vartheta}  \,\,\, F(\vartheta) \,,
	$$
    where $F(\theta)$ can be any nonlinear nonconvex function.
    
    (todo: insert link to example)

2. **Uncertainty Quantification**: given a set of data $d$, respectively the output of a model $f(x;\vartheta)$, a likelihood function $p(d|\vartheta)$,  and a prior probability density $p(\vartheta)$ Korali samples the posterior distribution
	$$
	p(\vartheta | d) = \frac{p(d | \vartheta) \, p(\vartheta)}{p(d)}\, .
	$$
    I.e. the output is a set of samples $\theta^{(k)}$ with $k \in \{1,...,N\}$ such that
    $$
		\vartheta^{(k)} \sim p(\theta | d).
    $$
    Note that $x$ are the input parameters of the computational model $f$.

    (todo: insert link to example)

After [installing](installation.md) the software, please have a look at the [examples](./examples/sampling.md) and learn how to run your own code.


## Notes

Please send questions and feedback to:

- martiser AT ethz.ch
- garampat AT ethz.ch
- wadaniel AT ethz.ch


<br><br><br><br><br><br>

!!! warning
    The software and the documentation page are under continuous development. New pages and new feature will be constantly added.
