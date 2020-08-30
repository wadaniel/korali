**************************
HierarchicalLatent
**************************

This is the lowlevel definition of hierarchical latent problems.
It is designed to be used internally.

The problem classes 'HierarchicalReference' and 'HierarchicalCustom' are wrappers for
this and provide a considerably simpler way to define variables.




Implementation Notes
---------------------

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

   However, the likelihood function - that is, :math:`p(\;data \;|\; latent \; variables\;)` - is implemented
   in the 'highlevel' problems, because they calculate the likelihood in different ways.
   This results in the following calling sequence (slightly simplified):

   solver
   --> highlevelProblem's :code:`evaluateLoglikelihood()`
   --> lowlevelProblem's :code:`evaluateLoglikelihood()`
   --> higlevelProblem's implementation of the log likelihood function
   --> user-defined computational models or log likelihood functions.
