import load_data
from _model import utils

import numpy as np


class ConditionalDistribution5():
  ''' Model 5:
     - multiple dimensions
     - multiple distribution types
     - latent variable coordinates are correlated
     - p(datapoint | latent) is still a normal distribution N(latent, sigma**2)
    '''

  def __init__(self):
    ''' Load the data from disk.
        self._p is an object that manages the data, and stores fixed parameter sigma, plus other information. '''
    self._p = load_data.PopulationData()

  def conditional_p(self, sample, points=None, internalData=False):

    latent_vars = sample["Latent Variables"]
    assert len(latent_vars) == self._p.nDimensions
    if internalData:
      assert points is None, "Points are handled internally"
      points = sample["Data Points"]
    else:
      assert points is not None

    sigma = self._p.sigma
    logp_sum = 0

    for point in points:
      logp = 0
      for dim in range(self._p.nDimensions):
        pt = point[dim]
        mean = latent_vars[dim]
        p = utils.univariate_gaussian_probability([mean], sigma, [pt])
        logp += np.log(p)
      logp_sum += logp

    sample["logLikelihood"] = logp_sum
