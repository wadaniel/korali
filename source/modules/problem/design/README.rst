**************************
Design
**************************

The design problem is considering information gain

.. math::

  U(s)=\int_{\mathcal{Y}} \int_{\mathcal{T}} \ln \frac{p(y \mid \vartheta, s)}{p(y \mid s)} p(y \mid \vartheta, s) p(\vartheta) \mathrm{d} \vartheta \mathrm{d} y

where :math:`y` denotes the measurements taken with design :math:`s` with parameters :math:`\vartheta`. For the numerical intergration of the integral we leverage the integration module, for the necessary model evaluations, the propagation module is used. In order to find the optimal experimental design

.. math::

  s^\star = \arg\max_s U(s)

we use the available optimization modules.