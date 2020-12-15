#
#   ! Run this from the tutorial's main directory, from 12.hierarchical.latent.variables - else you get path errors
#
#   Test whether the distribution parameterized by S, zeta and phi in model.py is correct or not.
#

import numpy as np
import pdb

from load_data import *
from model import *
from utils import *

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


def test_distribution_3():
  # Initialize the distribution
  distrib3 = ExampleDistribution3()

  d2_numberLatentVars = distrib3._p.nIndividuals
  d2_numberHyperparams = 1

  # Some value pairs:
  hyperparams = []  # simple list
  data = []  # list of lists
  latents = []  # list of lists
  _assignments = []  # list of lists, might stay unused
  sigmas = []
  omegas = []

  # ** The test inputs **
  hyperparams.append(1)  # (theta)
  latents.append([0.75])  # 1 individual
  data.append([0.5])
  _assignments.append([0])
  sigmas.append(0.5)
  omegas.append(0.5)

  hyperparams.append(6.5)  # (theta)
  latents.append([6.0, 6.8, 4.9])  # 3 individuals
  data.append([5.1, 7.3, 3.9])
  _assignments.append([0, 1, 2])
  sigmas.append(1.)
  omegas.append(0.5)

  hyperparams.append(-10)  # (theta)
  latents.append([-20, 5, -13, 0])  # 4 individuals
  data.append([-19.6, 5.25, -13.1, -0.2])
  _assignments.append([0, 1, 2, 3])
  sigmas.append(0.25)
  omegas.append(10)

  for i in range(len(hyperparams)):
    current_data = data[i]
    current_latents = latents[i]
    theta = hyperparams[i]
    sigma = sigmas[i]
    omega = omegas[i]
    p = np.product([
        univariate_gaussian_probability([l], sigma, [d])
        for d, l in zip(current_data, current_latents)
    ])
    p *= np.product([
        univariate_gaussian_probability([theta], omega, [l])
        for l in current_latents
    ])

    distrib3._p.reset_to(
        len(current_latents), sigma, omega, current_data, nSamplesEach=1)

    # /* Use the distributions S, zeta and phi functions to calculate the probability */
    k = {}  # fake korali sample
    k["Latent Variables"] = current_latents
    k["Hyperparameters"] = [theta]

    distrib3.S(k)
    distrib3.zeta(k)
    distrib3.phi(k)

    _zetaValue = k["zeta"]
    _sValues = k["S"]
    _phiValues = k["phi"]

    p_from_model_direct = np.exp(-_zetaValue + np.inner(_sValues, _phiValues))

    assert (np.abs(p - p_from_model_direct) < 0.1 * p)

  return True


if __name__ == '__main__':
  success = test_distribution_3()
  assert success
  print("OK")
