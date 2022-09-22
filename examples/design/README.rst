Integration
===========

In this tutorial we show how to determine an optimal **design** for a given function. The example is based on section 5.1. of https://arxiv.org/abs/1108.4146.

Problem Description
------------------- 

We are given the measurement model :math:`y(\vartheta,d)=g(\vartheta,d)+\epsilon=\vartheta^3 d^2 + \theta \exp(-|0.2-d|)+\epsilon` for :math:`\epsilon\sim U(0,1)`. We want to estimate the exptected information gain 

.. math::

  U(d)=\int_{\mathcal{Y}} \int_{\mathcal{T}} \ln \frac{p(y \mid \vartheta, d)}{p(y \mid d)} p(y \mid \vartheta, d) p(\vartheta) \mathrm{d} \vartheta \mathrm{d} y

in the parameter :math:`\vartheta\sim U(0,1)` of a measurement :math:`y` for a given choice of design :math:`d\in[0,1]`.

The Model
---------

Create a folder named `model`. Inside, create a file with name `model.py` which implements the model,

.. code-block:: python

        def model1D(s):
            theta = s["Parameters"]
            d = s["Designs"]
            res = theta * theta * theta * d * d + thetas * np.exp(-np.abs(0.2 - d))
            return res

Not, that this implements :math:`g(\vartheta,d)` and we do not include the stochastic error.


The Problem Type
----------------

Then, we set the type of the problem to `Design` and set the measurement model

.. code-block:: python

        e["Problem"]["Type"] = "Design"
        e["Problem"]["Model"] = model1D

The Variables
-------------

In this problem there is three variables: :math:`\vartheta`,  :math:`d`, and :math:`y`. We want to regard designs :math:`d\in[0,1]` at :math:`101` equidistant points. For the parameter and measurements we take :math:`10^5` samples :math:`\vartheta^{(i)}\sim U(0,1)` and :math:`y^{(i,j)}(d)=g(\vartheta^{(i)},d)+\epsilon^{(j)}`. This is reflected in the configuration of the variables as follows:

.. code-block:: python

        e["Distributions"][0]["Name"] = "Uniform"
        e["Distributions"][0]["Type"] = "Univariate/Uniform"
        e["Distributions"][0]["Minimum"] = 0.0
        e["Distributions"][0]["Maximum"] = 1.0

        e["Variables"][0]["Name"] = "design"
        e["Variables"][0]["Number Of Samples"] = 101
        e["Variables"][0]["Lower Bound"] = 0.0
        e["Variables"][0]["Upper Bound"] = 1.0
        e["Variables"][0]["Distribution"] = "Grid"
        
        e["Variables"][1]["Name"] = "theta"
        e["Variables"][1]["Lower Bound"] = 0.0
        e["Variables"][1]["Upper Bound"] = 1.0
        e["Variables"][1]["Number Of Samples"] = 1e5
        e["Variables"][1]["Distribution"] = "Uniform"
        
        e["Variables"][2]["Name"] = "measurement"
        e["Variables"][2]["Number Of Samples"] = 1e5

The Solver
----------
We choose the solver `Designer`, don't set the execution per generation, to have the evaluations of the model performed in one generation. Furthermore we set the standard deviation for the measurement error

.. code-block:: python

        e["Solver"]["Type"] = "Designer"
        e["Solver"]["Measurement Sigma"] = 1e-4

Running
-------

Finally, we need to add a call to the run() routine to start the Korali engine.

.. code-block:: python

    k.run(e)

In order to launch the example we use `python3 ./run-quadrature-integration.py`. Per default, the results are saved in the folder `_korali_result/`.
