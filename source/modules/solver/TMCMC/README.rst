*****************************************************
Transitional Markov Chain Monte Carlo
*****************************************************

This is the implementation of the *Transitional Markov Chain Monte Carlo*
algorithm, as published in `Ching2007 <https://ascelibrary.org/doi/abs/10.1061/%28ASCE%290733-9399%282007%29133%3A7%28816%29>`_.

TMCMC avoids sampling from difficult target probability densities
(e.g. posterior distributions in a Bayesian inference problem) but samples from
a series of intermediate PDFs that converge to the target PDF.
This technique is also known as Sampling Importance Resampling in the Bayesian community.

**Usage:**

.. code-block:: python

  k["Solver"] = "TMCMC"
  
**Requirements:**

+ The *Population Size* used at every generation needs to be defined.
+ The *Problem* needs to be of type Bayesian.

## Configuration

### Solver Settings
 
### Variable-Specific Settings

### Termination Criteria

## Plotting

Here we explain technical details of the **TMCMC** result plot.

The `python3 -m korali.plotter` command plots the distribution of the samples at every
generation. The samples are read from the json-files stored in the output
directory (`/_korali_result/`).

A plot of the samples obtained after the final generation of TMCMC
function is given below. Here, the target function is the exponential of the
negative of the 2-dimensional [Rosenbrock](https://en.wikipedia.org/wiki/Rosenbrock_function)
function.

![figure](sampling_rosenbrock.png)

**Diagonal Plots**. The diagonal plots show the marginal probability densities of the samples for
each variable. Note that the indices of the vertical axes correspond to the
upper and lower triangle plot and not to the diagonal plots.

**Upper Triangle**. In the plots in the upper triangle we see the actual realization of the samples.
The axes represent the dimensions, respectively the objective variables,
of the problem and we show a two-dimensional plot for every variable pair.

**Lower Triangle**. The lower triangle shows the probability density of the samples for each variable pair.
The density is approximated through a smoothening operator applied to the number
of samples that can be found in a given area.
