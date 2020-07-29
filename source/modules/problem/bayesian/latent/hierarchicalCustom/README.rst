************************************************************
Hierarchical Latent-Variable Problem with Custom Likelihood
************************************************************

Hierarchical latent problems impose a specific hierarchical
form on the total likelihood:

.. math::
  p( d, \theta  | \psi ) = \prod_{i=0}^N \left( \prod_{j=0}^{n_i} p(d_{i,j} | \theta_i) \right)
  \cdot p(\theta_i | \psi)


where

- :math:`d` is the data, where we have a varying number :math:`n_i` of data points :math:`d_{i,j}` for each
  'individual' :math:`i`

  (Note: The data can be entirely handled by the user. No data points are also possible.)
- Vectors :math:`\theta_i` are latent variables, one per 'individual' :math:`i`
- :math:`\psi` are a number of hyperparameters.

In the *Custom* problem class, you are completely free in the construction of a likelihood function.

Please refer to the corresponding tutorial / example for further explanation, such
as the form that we impose on :math:`p(\theta | \psi)`.