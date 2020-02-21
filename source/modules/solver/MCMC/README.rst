*****************************************************
Delayed Rejection Adaptive Metropolis Algorithm
*****************************************************

This is an implementation of the *Delayed Rejection Adaptive Metropolis* algorithm,
as published in [Haario2006](https://link.springer.com/article/10.1007%2Fs11222-006-9438-0).
This solver can also be configured to run the standard *Metropolis Hastings* method.

**Usage:**

.. code-block:: python

   k["Solver"]["Type"] = "MCMC"
  
**Plotting**

Here we explain technical details of the **MCMC** result plot.

The `python3 -m korali.plotter` command plots the distribution of the samples at every
generation. The samples are read from the json-files stored in the output
directory (`/_korali_result/`).

A plot of the samples obtained after the final gerneration of MCMC
function is given below. Here, the target function is the exponential of the
negative of the 2-dimensional [Rosenbrock](https://en.wikipedia.org/wiki/Rosenbrock_function)
function.

![figure](sampling_rosenbrock.png)

**Diagonal Plots**. The diagonal plots show the marginal probability densities of the samples for
each variable. Note that the indices of the vertical axes correspond to the
upper and lower triangle plot and not to the diagonal plots.

**Upper Triangle**. In the plots in the upper triangle we see the actual realization of the samples.
The axes represent the dimensions, respectiely the objective variables,
of the problem and we show a two-dimensional plot for every variable pair.

**Lower Triangle**. The lower triangle shows the probability density of the samples for each variable pair.
The density is approximated through a smoothening operator applied to the number
