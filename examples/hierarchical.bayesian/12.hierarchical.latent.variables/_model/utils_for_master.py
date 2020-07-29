import numpy as np
import copy
#
#  !! Use this instead of utils.py !! when you push things to master !!
#
#


def univariate_gaussian_probability(mu, sigma: float, point):
  point = np.array(point)
  mu = np.array(mu)
  n = len(mu)
  assert (len(mu) == len(point))
  squared_distance = np.inner(mu - point, mu - point)
  denominator = (np.sqrt(2.0 * np.pi) * sigma)**n
  probability = np.exp(-squared_distance / (2.0 * sigma**2)) / denominator
  return probability


def inv_logit(z):
  return 1. / (1. + np.exp(-z))


def logit(p):
  assert 0 <= p < 1
  return np.log(p / (1. - p))


def transform_to_z(point, d_normal, d_lognormal, d_logitnormal):
  ''' Assumes coordinates in :param point are ordered: normal, then lognormal, then logitnormal.'''
  z = np.zeros_like(point)
  eps = 1.e-15  # cannot be smaller than around 1.e-17
  d = d_normal + d_lognormal + d_logitnormal
  z[:d_normal] = point[:d_normal]
  z[d_normal:d_normal + d_lognormal] = [
      (np.log(p) if p > 0 else np.log(eps))
      for p in point[d_normal:d_normal + d_lognormal]
  ]
  z[d - d_logitnormal:] = [(logit(p) if (0 <= p < 1) else
                            (logit(eps) if (p < 0) else logit(1 - eps)))
                           for p in point[d - d_logitnormal:]]
  return z


def transform_from_z(z, d_normal, d_lognormal, d_logitnormal):
  point = np.zeros_like(z)
  d = d_normal + d_lognormal + d_logitnormal
  point[:d_normal] = z[:d_normal]
  point[d_normal:d_normal +
        d_lognormal] = [np.exp(p) for p in z[d_normal:d_normal + d_lognormal]]
  point[d - d_logitnormal:] = [inv_logit(p) for p in z[d - d_logitnormal:]]
  return point


def test_transform_to_z():
  for i in range(30):
    d_normal, d_lognormal, d_logitnormal = np.random.choice(
        list(np.arange(10)), 3)
    d = d_normal + d_lognormal + d_logitnormal
    z = np.random.uniform(-10, 10, (d,))
    point_0 = z[:d_normal].tolist()
    point_1 = [np.exp(z_) for z_ in z[d_normal:d_normal + d_lognormal]]
    point_2 = [inv_logit(z_) for z_ in z[d - d_logitnormal:]]
    point = np.array(point_0 + point_1 + point_2)
    if not np.allclose(
        z,
        transform_to_z(point, d_normal, d_lognormal, d_logitnormal),
        rtol=1.e-8):
      print("aaaah")
    assert np.allclose(
        z,
        transform_to_z(point, d_normal, d_lognormal, d_logitnormal),
        rtol=1.e-8)


#
#
#
#  REMOVE:
# def generate_variable(transf, experiment, idx, name, distribs, initial=None):
#   ''':param experiment: korali-experiment
#         :param idx: The variable index of the variable to create
#         :distribs: A dict mapping distribution types to prior distribution names'''
#   if transf == 0:  # Normal
#     experiment["Variables"][idx]["Name"] = "(Normal) " + name
#     experiment["Variables"][idx][
#         "Initial Value"] = -5. if initial is None else initial
#     experiment["Variables"][idx]["Bayesian Type"] = "Latent"
#     experiment["Variables"][idx]["Latent Variable Distribution Type"] = "Normal"
#     experiment["Variables"][idx]["Prior Distribution"] = distribs[
#         "Normal"]  # not used, but required
#   elif transf == 1:  # lognormal
#     experiment["Variables"][idx]["Name"] = "(Log-normal) " + name
#     experiment["Variables"][idx][
#         "Initial Value"] = 5. if initial is None else initial
#     experiment["Variables"][idx]["Bayesian Type"] = "Latent"
#     experiment["Variables"][idx][
#         "Latent Variable Distribution Type"] = "Log-Normal"
#     experiment["Variables"][idx]["Prior Distribution"] = distribs[
#         "Log-Normal"]  # not used, but required
#   elif transf == 2:  # probit-normal
#     raise NotImplementedError("Probit-normal variables not yet implemented")
#     experiment["Variables"][idx]["Name"] = "(Probit-normal) " + name
#     experiment["Variables"][idx][
#         "Initial Value"] = 999999. if initial is None else initial
#     experiment["Variables"][idx]["Bayesian Type"] = "Latent"
#     experiment["Variables"][idx][
#         "Latent Variable Distribution Type"] = "Probit-Normal"
#     experiment["Variables"][idx]["Prior Distribution"] = distribs[
#         "Probit-Normal"]  # not used, but required
#   elif transf == 3:  # logit-normal
#     experiment["Variables"][idx]["Name"] = "(Logit-normal) " + name
#     experiment["Variables"][idx][
#         "Initial Value"] = 0.5 if initial is None else initial
#     experiment["Variables"][idx]["Bayesian Type"] = "Latent"
#     experiment["Variables"][idx][
#         "Latent Variable Distribution Type"] = "Logit-Normal"
#     experiment["Variables"][idx]["Prior Distribution"] = distribs[
#         "Logit-Normal"]  # not used, but required
#   else:
#     raise ValueError("Only values 0 - 3 code for latent variable types")
#   return None
