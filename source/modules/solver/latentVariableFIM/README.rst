******************************************************************
Fischer Information Matrix estimation for latent variable problems
******************************************************************

This solver estimates the observed Fischer information matrix (FIM) for a "Bayesian/Latent" problem.
It currently can be used with `HierarchicalReference` and `HierarchicalCustom` problem
classes.

The observed Fischer information matrix can for example be used to approximate a parameter
probability distribution around an hyperparameter estimate. It is defined as

.. math::

  I = (I_{i,j})_{i,j=1}^{N}  \\
  I_{i,j}(\psi) = -\partial_{\psi_i}\partial_{\psi_j} \left(  log(p(d | \psi))  \right)

Here, :math:`d` is the observed data and :math:`\psi` are hyperparameters - the probability :math:`p(d | \psi)`
here refers to the marginal distribution, with latent variables integrated out.
I is a matrix of dimension :math:`N x N`, with :math:`N` the number of hperparameters.
We follow the approximation method described in [1] in section 9.4.1.

..
   *Developer note:*
   Although created to be also extensible to the last 'latent variable' problem class,
   `Latent/Exponential`, the FIM solver has become somewhat specific to the structure
   of the two 'higlevel' hierarchical latent variable problems.
   To add support for `Latent/Exponential`, differentiate between it and hierarchical in the
   solver's initialization (both functions), add user-defined gradient and Hessian calculation
   functions to `Latent/Exponential`, add internal parameters 'Latent Space Dimensions' and
   'Number Hyperparameters' to it, and then
   convert all uses of `sample["Mean"]`, `sample["Covariance Cholesky Decomposition"]`, etc to
   more generically just passing sample["Hyperparameters"] - mean and covariance are specific to
   hierarchical problems.

Solver parameters include values for the hyperparameters, for which to estimate the FIM.

TODO: Please also see the examples in :code:`examples/hierarchical.bayesian/latent.variables`:
- :code:`estimate-FIM-logistic.py`
- :code:`estimate-FIM-normal.py`
and the documentation for :code:`estimate-FIM-normal.py`.




Implementation Notes
---------------------
There are a few things that do not follow a good style practice and/or might be confusing.

1. Interaction between :code:`hierarchicalReference` / :code:`hierarchicalCustom` classes and the Lowlevel problem class:

   This is an issue not about FIM calculation, but about the problem classes:
   To simplify variable creation, the 'highlevel' problem classes :code:`hierarchicalReference`
   and :code:`hierarchicalCustom` create a second problem of type :code:`hierarchicalLowlevel`.
   This lowlevel problem has a complete
   set of variables (all hyperparameters and all latent variables), while the 'highlevel'
   problems are created with only one set of latent variables for a single individual.
   To ensure consistency, the highlevel classes' functions all call the corresponding function from
   the lowlevel problem. So, the lowlevel class contains the actual implementation, while the highlevel
   classes are a wrapper for it.

   However, the likelihood function - that is, :math:`p(data | latent variables)` - is implemented
   in the 'highlevel' problems, because they calculate the likelihood in different ways.
   This results in the following calling sequence (slightly simplified):

   solver
   --> highlevelProblem's :code:`evaluateLoglikelihood()`
   --> lowlevelProblem's :code:`evaluateLoglikelihood()`
   --> higlevelProblem's implementation of the log likelihood function
   --> user-defined computational models or log likelihood functions.

2. How parallelism is implemented in the sampling part of FIM calculation:
   There are two loops, one running within the other, that provide opportunity for parallelization:

   First, a user-defined number of chains are run that are independent from each other and
   can be run in parallel.

   Second, conditional log-likelihood calculations in both :code:`hierarchicalReference` and
   :code:`hierarchicalCustom` sum the conditional log-likelihoods for each individual.
   The log-likelihoods of different individuals are independent from each other (they depend
   on different user-defined computational models or log-likelihood functions).
   Thus, we could parallelize the individuals' calculations.

   As nested parallelism is not yet possible, this leaves the following options:

   a) Currently implemented: Parallelize the outer loop, over sampling chains.
      *Disadvantage:* Very slow for experiments with many (>=100) individuals.

   b) Parallelize the inner loop, over individuals.
      *Disadvantage:* Bad style, as it moves parallelization from solver to problem.
      It's also slightly slower than a).

   c) Parallelize the outer loop as in a). Change the problem implementation so that
      the user-defined computational models (for :code:`hierarchicalReference`) and
      log likelihood functions (for :code:`hierarchicalCustom`) are combined into one
      function or model that has to return a list of results, one per individual.
      Here, the user would be responsible for parallelizing (or vectorizing) the
      inner loop - something that might be nontrivial in Python.

   d) Rewrite sampling entirely: Since hyperparameters are fixed, the latent variable
      vector for one individual can be considered independently of all other latent variables.
      (Or am I overlooking something here?) Thus it should be possible to have L
      numbers of chains *per individual*. In this way one could sample latent variables
      for each individuals independently, and the two loops would be combined into one
      that iterates both over chains and individuals. (Again, am I overlooking something?)



[1] Lavielle, Marc. Mixed effects models for the population approach: models, tasks, methods and tools. CRC press, 2014.
