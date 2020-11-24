#!/usr/bin/env python
import numpy as np
import math

def model_0(s):
  v = s["Parameters"][0]
  s["logP(x)"] = -0.5 * v * v
  s["grad(logP(x))"] = [-v]
  s["H(logP(x))"] = [[-1.0]]

def model_1(s):
  v = np.array(s["Parameters"])
  dim = len(v)

  cov = np.array([[1.0, 0.0], [0.0, 5.0]], dtype=float)
  cov_inv = np.linalg.inv(cov)

  mu = np.zeros(dim, dtype=float)
  v_centred = v - mu

  s["logP(x)"] = -0.5 * np.matmul(np.matmul(v_centred.T, cov_inv), v_centred)

  s["grad(logP(x))"] = (-np.matmul(cov_inv, v_centred)).tolist()

  # print((-cov_inv).tolist())
  s["H(logP(x))"] = (-cov_inv).tolist()


def model_1b(s):
  v = np.array(s["Parameters"])
  dim = len(v)

  cov = np.array([[2.25, 0.0], [0.0, 0.0001]], dtype=float)
  cov_inv = np.linalg.inv(cov)

  mu = np.zeros(dim, dtype=float)
  v_centred = v - mu

  s["logP(x)"] = -0.5 * np.matmul(np.matmul(v_centred.T, cov_inv), v_centred)

  s["grad(logP(x))"] = (-np.matmul(cov_inv, v_centred)).tolist()

  # print((-cov_inv).tolist())
  s["H(logP(x))"] = (-cov_inv).tolist()

def model_2(s):
  v = np.array(s["Parameters"])
  dim = len(v)

  cov = np.array([[0.25, 0.3], [0.3, 1.0]], dtype=float)
  cov_inv = np.linalg.inv(cov)

  mu = np.ones(dim, dtype=float)
  v_centred = v - mu
  
  s["logP(x)"] = -0.5 * np.matmul(np.matmul(v_centred.T, cov_inv), v_centred)

  s["grad(logP(x))"] = (-np.matmul(cov_inv, v_centred)).tolist()

  s["H(logP(x))"] = (-cov_inv).tolist()


def model_3(s):
  v = np.array(s["Parameters"])
  dim = len(v)

  cov = np.identity(dim)
  cov_inv = np.linalg.inv(cov)

  mu = np.zeros(dim, dtype=float)
  v_centred = v - mu

  s["logP(x)"] = -0.5 * np.matmul(np.matmul(v_centred.T, cov_inv), v_centred)

  s["grad(logP(x))"] = (-np.matmul(cov_inv, v_centred)).tolist()

  s["H(logP(x))"] = (-cov_inv).tolist()


# log exponential with mean 4
def lexponential(s):
  lam = 1. / 4.
  x0 = s["Parameters"][0]
  r = 0.0
  if (x0 < 0):
    r = -math.inf
  else:
    r = math.log(lam) - lam * x0
  s["logP(x)"] = r
  s["grad(logP(x))"] = [-lam]
  s["H(logP(x))"] = [[0.0]]

  
# log laplace with mean 4 and scale 1 (var 2)
def llaplace(s):
  x0 = s["Parameters"][0]
  mu = 4.0
  scale = 1
  r = -math.log(2.0 * scale) - abs(x0 - mu) / scale
  s["logP(x)"] = r
  s["grad(logP(x))"] = [-np.sign(x0 - mu) / scale]
  s["H(logP(x))"] = [[0.0]]

def lcauchy(s):
  x0 = 0.0
  x = s["Parameters"][0]
  gamma = 1
  s["logP(x)"] = 2.0 * math.log(gamma) - math.log(math.pi) - math.log((x - x0)**2 + gamma ** 2)
  s["grad(logP(x))"] = [-2.0 * (x - x0) / ((x - x0)**2 + gamma ** 2)]
  s["H(logP(x))"] = [[(2.0 * (x - x0) ** 2 - 2.0 * gamma ** 2) / (((x - x0) ** 2 + gamma**2)**2)]]


# helper
def lognormal(x, sdev):
    return -0.5*math.log(2*math.pi*sdev**2)-0.5*(x/sdev)**2

# funnel function from paper
def lfunnel(s):
  param = s["Parameters"]
  v = param[0]
  logp = lognormal(v, 9.0)
  for i in range(len(param)-1):
    logp += lognormal(param[i+1], math.exp(-v))

  
  s["logP(x)"] = logp
  #s["grad(logP(x))"] = 
  #s["H(logP(x))"] = 


