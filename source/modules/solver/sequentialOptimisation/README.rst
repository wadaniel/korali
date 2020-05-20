**************************
Sequential Optimisation
**************************

Sequential Optimisation performs optimisation

.. math::

  s^*=\arg\max U(s)

where the variable s can be written as a collection of values

.. math::

  s=(s_1,\dots,s_n)

and the objective function is defined for all n. The optimisation is performed sequentially

.. math::

  s^*_n=\arg\max_s U(s_1,\dots,s_{n-1},s)

