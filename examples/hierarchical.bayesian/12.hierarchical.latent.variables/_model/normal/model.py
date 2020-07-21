import numpy as np
import sys, os

scriptdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(scriptdir, '../'))
import load_data


def normalModel(x, theta):
  y = theta**2
  if np.any(np.isinf(y) | np.isnan(y)):
    y = 1e300
  return y

def normalModelFunction(sample, points):
  ''' :param points: Data points to calculate the function evaluation from.
                    Here, only used to find the number of reference evaluations for this individual'''
  N = len(points)
  latents = sample["Latent Variables"]
  assert len(latents) == 2
  theta = latents[0]
  sdev = latents[1]
  sdev = abs(sdev)

  referenceEvals = []
  for i in range(N):
    y = theta**2
    if np.any(np.isinf(y) | np.isnan(y)):
      y = 1e300
    referenceEvals.append(y)

  sample["Reference Evaluations"] = referenceEvals
  sample["Standard Deviations"] = [sdev] * N


class NormalConditionalDistribution():
  ''' Model 6:
        Data generation process: yi = f(xi, theta) + eps,
                where eps ~ N(0, alpha) (alpha is the standard deviation and is known from the model.)
            Everything is one-dimensional.
    '''

  def __init__(self):
    self._p = load_data.NormalData()

  def conditional_p(self, sample, points):

    latent_vars = sample["Latent Variables"]
    assert len(latent_vars) == self._p.nLatentSpaceDimensions == 2, f"Latent variable vector has wrong length. " \
                                                                    f"Was: {len(latent_vars)}, should be: {2}"
    logp_sum = 0

    for point in points:
      assert len(point) == 3, "Expected id, x and y values as 'data point'"
      x = point[1]
      y = point[2]
      fx = normalModel(x, latent_vars[0])
      sigma2 = latent_vars[-1]**2
      eps = 1e-10
      if self._p.error_model == "constant":
        err = (y - fx)**2
        det = sigma2
      elif self._p.error_model == "proportional":
        y2 = max(y**2, eps)
        err = (y - fx)**2 / y2
        det = sigma2 * y2
      else:
        raise ValueError(f"Unknown error model: {self._p.error_model}")

      if np.isinf(err) or np.isnan(err):
        logp_sum = -1.e200
      else:
        log2pi = 0.5 * np.log(2 * np.pi)
        logp = -log2pi - 0.5 * np.log(det) - 0.5 * err / sigma2
        logp_sum += logp

    sample["logLikelihood"] = logp_sum