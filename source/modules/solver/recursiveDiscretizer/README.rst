****************************
Recursive Discretizer
****************************

This solver approximates a continuous process by discretizing the parameter space, leading to a recurrence relation analog to the Hamilton–Jacobi–Bellman equation:

 .. math::

   J_{k}^{\ast} \left( \mathbf{x}_{n-k} \right) = \min_{\mathbf{u}_{n-k}} \left\{ \hat{f} \left( \mathbf{x}_{n-k}, \mathbf{u}_{n-k} \right) + J_{k-1}^{\ast} \left( \hat{g} \left( \mathbf{x}_{n-k}, \mathbf{u}_{n-k} \right) \right) \right\}
   
at the :math:`k`-th stage of :math:`n` equally spaced discrete time intervals, and where :math:`\hat{f}` and :math:`\hat{g}` denote discrete approximations to :math:`f` and :math:`\mathbf{g}`. This functional equation is known as the Bellman equation, which can be solved for an exact solution of the discrete approximation of the optimization equation. See: Kirk, Donald E. `Optimal Control Theory: An Introduction <https://books.google.com/books?id=fCh2SAtWIdwC&pg=PA94>`_

The solution is found by converting the discretized parameter space to a connected graph and finding the best policy by applying `Dijkstra's (shortest path) algorithm <https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm>`_. 