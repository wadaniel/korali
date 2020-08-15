==========================================
Latent Variable Problem
==========================================

In a Bayesian *latent variable* problem, the model includes a set of unobserved variables, in addition to the variables parameterizing the distribution that we want to estimate.

That is, we have a probability distribution of the following form:

.. math::
  p( d, z  | \psi )

Here, d are observed data points, z the latent variables, and :math:`\psi` the distribution parameters (aka hyperparameters).

Do not use this class directly, but instead one of the subclasses `Exponential`
and `HierarchicalCustom` or `HierarchicalReference`. An `Exponential`
problem makes fewer assumptions on the distribution - it assumes that it belongs to the
exponential family. A `HierarchicalCustom` or `HierarchicalReference` problem instead
assumes a hierarchical probability structure.
Please refer to their respective documentation.


Configuration
-------------


Variable-Specific Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each variable you define, please set

.. code-block:: python

    e["Variable"][your_index][ "Bayesian Type" ] = "Latent"
    # or
    e["Variable"][your_index][ "Bayesian Type" ] = "Hyperparameter"

to select what role this variable takes in the problem. Note that for `Hierarchical Latent`
problems, you can only declare latent variables and select their distribution; hyperparameters
will be inferred.
