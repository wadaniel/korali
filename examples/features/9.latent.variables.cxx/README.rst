--------------------------------------------
CXX Latent Variable Probability Optimization
--------------------------------------------


~~~~~~~~~~~~~~~~~~~
Under Construction!
~~~~~~~~~~~~~~~~~~~


C.4 - Running Korali with Compiled CXX Code, for optimizing parameters in a latent variable model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this tutorial we show how Korali can be used with c++.
Here we optimize a model with the solver `SA-EM` .
The problem is the sample application (ch. 6) of `[1] <https://www.tandfonline.com/doi/abs/10.1198/106186006X157469>`_ .
We want to find the parameters theta =  (...) that maximize the posterior in a Bayesian problem with latent variables.  

How to run the example
~~~~~~~~~~~~~~~~~~~~~~

Run the `Makefile` to compile the executables: In the command line, in the
c4-... sub-folder, type
.. code-block::

  make

*Make* finds our `Makefile` and reads the instructions inside how to make the first target listed there.
 This will will create an executable out of each of our .cpp files. To run one of them, type their name prepended 
 with `./`, for example,

.. code-block::

  ./run-saem

This should output information about the process and result of the optimization.


[1] Jank, Wolfgang. "Implementing and diagnosing the stochastic approximation EM algorithm." Journal of Computational and Graphical Statistics 15.4 (2006): 803-829.