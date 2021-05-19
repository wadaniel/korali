#!/usr/bin/env python3
import json
import numpy as np


def open_json(dirname, filename):
  path = '{0}/{1}'.format(dirname, filename)
  data = None
  with open(path) as f:
    data = json.load(f)
  return data


def checkMean(k, expectedMean, tol):
  samples = k["Solver"]["Sample Database"]

  mean = np.mean(samples)
  assert np.isclose(
      expectedMean, mean, atol=tol
  ), "Sample mean ({0}) deviates from expected ({1}) by more than {2}".format(
      mean, expectedMean, tol)


def checkStd(k, expectedStd, tol):
  samples = k["Solver"]["Sample Database"]

  std = np.std(samples)
  assert np.isclose(
      expectedStd, std, atol=tol
  ), "Sample standard deviation ({0}) deviates from expected ({1}) by more than {2}".format(
      std, expectedStd, tol)
