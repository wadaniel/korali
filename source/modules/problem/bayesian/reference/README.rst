*************************
Likelihood by Reference
*************************

In a *Bayesian Inference* problem, the defines and a prior probability density $p(\vartheta)$ for the problem variables, and the solver is applied to the of the posterior distribution:

.. math::

   p(\vartheta | d) = \frac{p(d | \vartheta) p(\vartheta)}{p(d)}

Additive Normal Likelihood
---------------------------

With a Additive Normal Likelihood, the computational model is assumed to be of the form :math:`f(x;\vartheta)`, where :math:`d` is a set of M given data points. The output of the model represents the values of the function at the given points for which Korali can build a likelihood function :math:`p(d|\vartheta)`, and a prior probability density :math:`p(\vartheta)`. 

Currently, Korali uses a Normal estimator for the error component of the likelihood calculation, using a statistical-type variable, :math:`sigma`:

.. math::

   p(d | \vartheta) = {\frac {1}{\sigma {\sqrt {2\pi }}}}e^{-{\frac {1}{2}}\left((x-\mu )/\sigma \right)^{2}}

	
