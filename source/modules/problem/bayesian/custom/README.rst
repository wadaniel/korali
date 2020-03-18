**************************
Custom Likelihood
**************************

In a *Bayesian Inference* problem, the defines and a prior probability density $p(\vartheta)$ for the problem variables, and the solver is applied to the of the posterior distribution:

 $$ p(\vartheta | d) = \frac{p(d | \vartheta) p(\vartheta)}{p(d)} $$

## Likelihood Models 

### Additive Normal Likelihood

Whereas with a *Additive Normal Likelihood*, the computational model is assumed to be of the form $f(x;\vartheta)$, where $d$ is a set of *M* given data points. The output of the model represents the values of the function at the given points for which Korali can build a likelihood function $p(d|\vartheta)$, and a prior probability density $p(\vartheta)$. 

Currently, Korali uses a Normal estimator for the error component of the likelihood calculation, using a statistical-type variable, *sigma*:

$$ p(d | \vartheta) = {\frac {1}{\sigma {\sqrt {2\pi }}}}e^{-{\frac {1}{2}}\left((x-\mu )/\sigma \right)^{2}} $$

With a *Custom Likelihood*, the function $p(d|\vartheta)$ is given directly by a user-defined model of the form $f:\mathbb{R}^N\rightarrow\mathbb{R}$, where $N$ is the number of variables.

