**************************************************************************
Hierarchical Latent-Variable Problem with Likelihood by Reference
**************************************************************************

Hierarchical latent problems impose a specific hierarchical
form on the total likelihood:

.. math::
  p( d, \theta  | \psi ) = \prod_{i=0}^N \left( \prod_{j=0}^{n_i} p(d_{ij} | \theta_i) \right) \cdot p(\theta_i | \psi)


where

- :math:`d` is the data, where we have a varying number :math:`n_i` of data points :math:`d_{i,j}` for each
  'individual' :math:`i`

  (Note: The data can be entirely handled by the user. Korali then makes no assumptions about the data.)
- Vectors :math:`\theta_i` are latent variables, one per 'individual' :math:`i`
- :math:`\psi` are a number of hyperparameters.

A hierarchical *Reference* problem additionally assumes that the data come from a functional relationship:
Every data point :math:`x` comes with a reference evaluation :math:`y`, where we assume
:math:`y = f(x, \theta) + \epsilon`. :math:`\epsilon` is a noise term. Its distribution is defined by the
*likelihood model* for the data.

Our data is composed of :math:`n_i` data points and reference evaluations per individual :math:`i`:

.. math::


  d = (x_{ij}, y_{ij})_{i\in I, j\in J_i}, \\
  where \; I = 1...N, J_i = 1...n_i, \;\; and\\
  we \; assume \;\;\; y_{ij} = f(x_{ij}, \theta_i) + \epsilon_{ij}.

The x values :math:`X = (x_{ij})_{i\in I, j\in J_i}` can also be omitted. This corresponds to a
computational model :math:`f(\theta_i)` that does not depend on additional data points :math:`x_{ij}`.

Usage
~~~~~

.. code-block:: python

    # example data
    number_individuals = 3
    my_refrence_data = [ [1.5, 1.75],
                          [2.0],
                          [2.25, 3.0] ]

    my_data_points =    [ [[1,1],[2,2]],
                          [[0,0]],
                          [[3,3], [4,4]] ]

    my_other_parameters = ... # we might have other individual paramters, such as the age of each individual

    e = korali.Experiment()
    e["Problem"]["Type"] = "Evaluation/Bayesian/Latent/HierarchicalReference"
    e["Problem"]["Likelihood Models"] = "Additive Normal"
    e["Problem"]["Latent Space Dimensions"] = ... # the number of latent variables per individual
    e["Problem"]["Reference Data"] = my_refrence_data # insert list of lists of reference evaluations

Then, we can define the computational models for each individual as follows - here, the data
is managed by the user:

.. code-block:: python

    f = lambda sample, points, other_params: ... # will use 'points' sample["Latent Variables"] and do something with it


    # f should calculate the function value f(x, theta), as well as a standard deviation
    # sigma = g(x, theta) or dispersion coefficient, depending on likelihood model.
    e["Problem"]["Computational Models"] = [
            (lambda index:
                (lambda sample: f(  sample,
                                    my_data_points[index],
                                    my_other_parameters[index] ) )
            )(i)
            for i in range(number_individuals)]

**Important note:** We need this complicated form, because a mere :code:`[lambda s: f(s, my_points[i]) for i in range(10)]`
**will not work**. (In the lambda expression, Python will capture :code:`i` by reference, meaning each :code:`i`
will be overwritten with the last value for :code:`i` - that's 2 here. See `here <https://stackoverflow.com/questions/6076270/lambda-function-in-list-comprehensions>`_
for more information.)

Alternatively, to let Korali handle the data points (first without additional parameters):

.. code-block:: python

    f = lambda sample: ... # will access sample["Data Points"] and sample["Latent Variables"] and do something with it

    e["Problem"]["Computational Models"] = [f] * number_individuals # Each function is the same. Here this is fine.
    e["Problem"]["Data Points"] = my_data_points

With additional parameters, we again need more convoluted lambda expressions:

.. code-block:: python

    f = lambda sample, other_params: ... # will access 'other_params', sample["Data Points"] and sample["Latent Variables"] and do something with it

    e["Problem"]["Computational Models"] = [
            (lambda index:
                (lambda sample: f(  sample,
                                    my_other_parameters[index] ) )
            )(i)
            for i in range(number_individuals)]

    e["Problem"]["Data Points"] = my_data_points


-----------------------

Please refer to the corresponding example for further explanation and complete usage examples.
