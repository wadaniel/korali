import numpy as np
import sys, os

scriptdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(scriptdir, '../'))
import utils
import load_data


def logisticModel(x, theta):
  assert len(theta) >= 3
  # logistic function
  f = np.exp(float(theta[2]) * x)  # can give inf if theta[2] is very large
  y = (theta[0] * theta[1] * f) / (theta[0] + theta[1] * (f - 1.))

  if np.isscalar(x):
    if np.isinf(y) or np.isnan(y):
      y = 1e300
  else:
    y[np.isinf(y) | np.isnan(y)] = 1e300
    # set inf or nan to something large. otherwise nlmefitsa complains -- not sure if necessary in python, too. we'll see.
  return y


def logisticModelFunction(sample, points=None, internalData=False):
  theta = sample["Latent Variables"]
  if internalData:
    points = sample["Data Points"]
    assert points is None, "No need to pass 'points' if internal data is set"
  else:
    assert points is not None, "Missing data points"
  assert len(theta) >= 4
  referenceEvals = []

  for x in points:
    assert len(x) == 1
    x = x[0]
    f = np.exp(float(theta[2]) * x)  # can give inf if theta[2] is very large
    y = (theta[0] * theta[1] * f) / (theta[0] + theta[1] * (f - 1.))

    if np.isscalar(x):
      if np.isinf(y) or np.isnan(y):
        y = 1e300
    else:
      y[np.isinf(y) | np.isnan(y)] = 1e300
      # set inf or nan to something large. otherwise nlmefitsa complains -- not sure if necessary in python, too. we'll see.
    referenceEvals.append(y)

  sdev = theta[3]
  sample["Reference Evaluations"] = referenceEvals
  sample["Standard Deviations"] = [sdev] * len(points)


class LogisticConditionalDistribution():
  ''' Model 7:
        Data generation process: yi = f(xi, theta[:3]) + eps,
                where eps ~ N(0, theta[3]) ( theta[3] is the standard deviation)
            Everything is one-dimensional.

    '''

  def __init__(self):
    self._p = load_data.LogisticData()

  def conditional_p(self, sample, points=None, internalData=False):

    latent_vars = sample["Latent Variables"]
    assert len(latent_vars) == self._p.nLatentSpaceDimensions
    if internalData:
      points = sample["Data Points"]
      assert points is None, "Points are handled internally"
    else:
      assert points is not None

    logp_sum = 0

    for point in points:
      assert len(point) == 3, f"Latent variable vector has wrong length. " \
                          f"Was: {len(latent_vars)}, should be: {2}"
      x = point[1]
      y = point[2]
      fx = logisticModel(x, latent_vars[:-1])
      sigma2 = latent_vars[-1]**2
      eps = 1e-10
      if self._p.error_model == "constant":
        try:
          err = (y - fx)**2
        except OverflowError as e:
          err = 1e200
        det = sigma2
      elif self._p.error_model == "proportional":
        y2 = max(y**2, eps)
        try:
          err = (y - fx)**2 / y2
        except OverflowError as e:
          err = 1e200
        det = sigma2 * y2
      else:
        raise ValueError(f"Unknown error model: {self._p.error_model}")

    if np.isinf(err) or np.isnan(err):
       logp_sum = -1.e200
    else:
      log2pi = 0.5 * np.log(2 * np.pi)
      if (sigma2 == 0):
        logp = -np.inf
      else:
        logp = -log2pi - 0.5 * np.log(float(det)) - 0.5 * err / sigma2
      logp_sum += logp

    sample["logLikelihood"] = logp_sum