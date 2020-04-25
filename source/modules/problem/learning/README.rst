****************************
Learning
****************************

This module describes  `Control Theory <https://en.wikipedia.org/wiki/Control_theory>`_, problems:

This problem finds an admissible control :math:`\mathbf{u}^{\ast}` which causes the system :math:`\dot{\mathbf{x}}(t) = \mathbf{g} \left( \mathbf{x}(t), \mathbf{u}(t), t \right)` to follow an admissible trajectory :math:`\mathbf{x}^{\ast}` on a continuous time interval :math:`t_{0} \leq t \leq t_{1}` that minimizes a **cost function**

.. math::

   J = b \left( \mathbf{x}(t_{1}), t_{1} \right) + \int_{t_{0}}^{t_{1}} f \left( \mathbf{x}(t), \mathbf{u}(t), t \right) \mathrm{d} t 
   
The solution to this problem is an optimal control law or policy :math:`\mathbf{u}^{\ast} = h(\mathbf{x}(t), t)`, which produces an optimal trajectory :math:`\mathbf{x}^{\ast}` and an optimized reward function :math:`J^{\ast}`.

This module allows defining the cost function, which receives a sample containing a full proposed **policy**. Constraints to policies can be enforced by assigned a *-Infinite* cost to them.

