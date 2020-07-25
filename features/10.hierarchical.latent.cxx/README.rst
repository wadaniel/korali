===============================================================================
12 - Optimizing parameters in a hierarchical latent variable model, using SA-EM
===============================================================================

Here, we optimize hierarchical latent variable models. As in tutorial 11, we aim to find the parameters for a
probability distribution, that maximize the likelihood of a set of observed points :math:`X = \{\mathbf{x}_i\}_{i=1...M}`,
:math:`\mathbf{x}_i \in \mathbb{R}^n`. We first use the standard version of the `SA-EM` solver, and then show the use of
specialized `SA-EM` for hierarchical models (`hSAEM`).

As hierarchical model, we consider a probability distribution of the form (following [1]):


.. math::
    p(\mathbf{x}_i, \mathbf{\theta}_i | \mathbf{\psi})  = p( \mathbf{x}_i | \mathbf{\theta}_i ) \cdot q( \mathbf{\theta}_i | \mathbf{\psi})

with restricted prior :math:`q`. Each i indexes one 'individuum' from a population. Multiple datapoints might exist for a single individuum.

**The prior** :math:`q` **:** We assume the latent variables :math:`\mathbf{\theta}_i` can be transformed
into (or originate from) multinomial normal distributed variables
:math:`\mathbf{f}(\mathbf{z}_i) = \mathbf{\theta}_i`. That is,

.. math::

    y_i \; \sim \;  p(y_i | \mathbf{f}(\mathbf{z}_i))  \\
    \mathbf{z}_i \; \sim \; \mathcal{N}(\mathbf{\beta}, \mathbf{\Omega})

To this end, the user can choose between latent variables :math:`\theta_i` whose coordinates follow normal, log-normal or logit-normal
distributions. Log-normal means :math:`log(\theta) \; \sim \; \mathcal{N}(a, b)`, and logit-normal similarly means
:math:`logit(\theta) \; \sim \; \mathcal{N}(a, b)`, where :math:`logit(\theta) = log\left({\theta}/{(1 - \theta)}\right)`.
See also example 2 below.

Normal and log-normal distributions describe distributions in many different disciplines `[2] <https://stat.ethz.ch/~stahel/lognormal/bioscience.pdf>`_.

--------------
Two Examples
--------------

1. **Simple population example:**
   We first follow the simple example of **section 9.2.5 of Lavielle's book [1]**. All distributions are normal which makes
   sampling distribution and optimization analytically solvable [1].
   The model is:

   .. math::

      x_i \; \sim \; \mathcal{N}(\theta_i, \sigma^2) \; \text{for} \; 1 \leq i \leq N  \\
      \theta_i \; \sim \; \mathcal{N}(\beta, \omega)

   We have one sample per "individuum" i. Note that our notation is reversed to that of the book, we use :math:`\theta`
   for latent variables and :math:`\psi` for hyperparameters.

2. **Mixed prior distributions:**
    Here we extend the previous example to
     -  arbitrary numbers of dimensions
     -  all three types of prior distributions: Normal, log-normal, and logit-normal
     -  different numbers of measurements for each individual.

    The resulting probability distribution is:

   .. math::

      \mathbf{x}_{ij} \; \sim \; \mathcal{N}(\mathbf{\theta}_i, \;\mathbf{I}\cdot\sigma^2) \;\; \text{for} \;\; 1 \leq i \leq N, \; 1 \leq j \leq n_i  \\
      \mathbf{\theta}_i = \mathbf{f}(\mathbf{z}_i) \\
      \mathbf{z}_i \; \sim \; \mathcal{N}(\mathbf{\beta}, \mathbf{\Omega})

   That is, the data is again normally distributed around each individual's latent variable :math:`\mathbf{\theta}_i`, which is
   now vector-valued and whose transformed entries follow a multivariate normal distribution with mean :math:`\mathbf{\beta}`
   and covariance matrix :math:`\mathbf{\mathbf{\Omega}}`.
   The inverse transform :math:`\mathbf{h}` is the identity in each normal coordinate of :math:`\mathbf{\theta}_i`, while
   for log-normal coordinates it's the exponential and for logit-normal coordinates the sigmoidal function:

   :math:`\mathbf{h}(\mathbf{z})_k = h(z_k)`, with:

   - :math:`h(z_k) = z_k` for normal :math:`\theta_k`,
   - :math:`h(z_k) = exp(z_k)` for log-normal :math:`\theta_k`, and
   - :math:`h(z_k) = \frac{1}{1 + exp(-z_k)}` for logit-normal :math:`\theta_k` (sigmoid is the inverse logit).


Data for both examples can be generated with different parameters using the script `model/generate_data.py`.
This will also write detailed information about the second example into `_model/data_advanced_info.txt`. Please
refer to this for an estimate how difficult the generated data will be to estimate - it generates
a number of proxies for the optimizers. These can be compared to the true / original parameter values.



TODO: Describe how to run both examples and how to re-generate data and
read the information files. EDIT: Do that for the python example only.



References
==========

[1] Lavielle, Marc. Mixed effects models for the population approach: models, tasks, methods and tools. CRC press, 2014.

`[2] <https://stat.ethz.ch/~stahel/lognormal/bioscience.pdf>`_ Limpert, Stahel, Abbt. Log-normal Distributions across the Sciences: Keys and Clues. BioScience May 2001.

