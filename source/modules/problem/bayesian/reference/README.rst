*************************
Likelihood by Reference
*************************

In a *Bayesian Inference* problem, the user defines and a prior probability density :math:`p(\vartheta)` for the problem variables and a likelihood function :math:`p(d | \vartheta)`, and the solver is applied to the of the posterior distribution:

.. math::

   p(\vartheta | d) = \frac{p(d | \vartheta) p(\vartheta)}{p(d)}


The following likelihood functions are available in Korali:


Normal
------


.. math::

   p(d | \vartheta) = {\frac {1}{\sigma {\sqrt {2\pi }}}}e^{-{\frac {1}{2}}\left((x-\mu )/\sigma \right)^{2}}


Positive Normal
---------------

The *Normal* likelihood truncated at 0.


StudentT
--------


.. math::

   p(d | \vartheta) = {\frac {\Gamma((n+1)/2)}{{\sqrt {n\pi} \Gamma(n/2)}}}(1+d^2/n)^{-(n+1)/2}

where :math:`n` is refered to as Degrees Of Freedom.


Positive StudentT
-----------------

The *StudentT* likelihood truncated at 0.


Poisson
-------


.. math::

   p(d | \vartheta) = {\frac {\lambda^d e^{-\lambda} }{d!}}

where :math:`\lambda` is the mean.


Negative Binomial
-----------------


.. math::

   p(d | \vartheta) = {d+r-1\choose d} p^r (1-p)^d

where :math:`p` is the success probability and :math:`r` is the dispersion parameter.
