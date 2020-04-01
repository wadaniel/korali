****************************
Dynamic Programming
****************************

This module describes Dynamic Programming problems within `Control Theory <https://en.wikipedia.org/wiki/Control_theory>`_, as described in `https://en.wikipedia.org/wiki/Dynamic_programming <https://en.wikipedia.org/wiki/Dynamic_programming>`_:

In Control Theory, a typical problem is to find an admissible control :math:`\mathbf{u}^{\ast}` which causes the system :math:`\dot{\mathbf{x}}(t) = \mathbf{g} \left( \mathbf{x}(t), \mathbf{u}(t), t \right)` to follow an admissible trajectory :math:`\mathbf{x}^{\ast}` on a continuous time interval :math:`t_{0} \leq t \leq t_{1}` that minimizes a **cost function**

.. math::

   J = b \left( \mathbf{x}(t_{1}), t_{1} \right) + \int_{t_{0}}^{t_{1}} f \left( \mathbf{x}(t), \mathbf{u}(t), t \right) \mathrm{d} t 
   
The solution to this problem is an optimal control law or policy :math:`\mathbf{u}^{\ast} = h(\mathbf{x}(t), t)`, which produces an optimal trajectory :math:`\mathbf{x}^{\ast}` and an optimized loss function :math:`J^{\ast}`. The latter obeys the fundamental equation of dynamic programming:

.. math::

   - J_{t}^{\ast} = \min_{\mathbf{u}} \left\{ f \left( \mathbf{x}(t), \mathbf{u}(t), t \right) + J_{x}^{\ast \mathsf{T}} \mathbf{g} \left( \mathbf{x}(t), \mathbf{u}(t), t \right) \right\}

This module allows defining the cost function, which receives a sample containing a full proposed **policy**. Constraints to policies can be enforced by assigned an *Infinite* cost to them.

