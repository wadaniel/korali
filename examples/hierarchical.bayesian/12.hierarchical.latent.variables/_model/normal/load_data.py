import numpy as np
import os

def extr(line, toint=True, tofloat=False, delim=" "):
  ''' Helper for extracting file contents. '''
  result = line.strip("\n ").split(delim)
  if tofloat:
    result = [float(r) for r in result]
  elif toint:
    result = [int(r) for r in result]
  return result


class NormalData():

  def __init__(self, datafile=None):
    self.nIndividuals = None
    self.nDataTotal = None
    self.nDataDimensions = 1
    self.nLatentSpaceDimensions = None
    self.dNormal = None
    self.dLognormal = None
    self.dLogitnormal = None

    self.error = "ind"  # only 'individual' error parameters are possible
    self.error_model = "constant"  #  "proportional" might also work but is untested

    self.nSamplesEach = []
    self.data = []
    if datafile is None:
      filename = '_data/normal/all_data.txt'
    else:
      filename = os.path.join('_data/normal/', datafile)
    delimiterIn = '\t'
    print(f"Loading data from {filename} ... \n")
    with open(filename, "r") as fd:
      self.column_names = fd.readline().strip("\n ").split(delimiterIn)
      lines = fd.readlines()
      data = [extr(line, tofloat=True, delim=delimiterIn) for line in lines]
      data = np.array(data)
      uid = np.unique(data[:, 0])
      self.nIndividuals = len(uid)
      self.nDataTotal = len(data)
      self.nSamplesEach = np.zeros((self.nIndividuals))
      for i in range(self.nIndividuals):
        self.nSamplesEach[i] = np.sum(data[:, 0] == i)
      assert np.sum(self.nSamplesEach) == len(data)

    for i in range(self.nIndividuals):
      self.data.append(data[data[:, 0] == i])

    self.data = np.array(self.data)
    self.beta = [2, 2]
    self.omega = np.eye(2)
    self.Nmp = len(self.beta) - 1
    self.N = len(self.beta)
    self.nLatentSpaceDimensions = len(self.beta)
    self.omega_chol = np.linalg.cholesky(self.omega)
    self.sigma = 1 * np.eye(self.N)

    self.transf = 0
    self.err_transf = 1
    self.dNormal = np.sum(self.transf == 0) + np.sum(self.err_transf == 0)
    self.dLognormal = np.sum(self.transf == 1) + np.sum(self.err_transf == 1)
    self.dProbitnormal = np.sum(self.transf == 2) + np.sum(self.err_transf == 2)
    self.dLogitnormal = np.sum(self.transf == 3) + np.sum(self.err_transf == 3)
    assert self.dProbitnormal == 0, "Probitnormal variables not yet implemented"

  #
  # def reset_to(self, nIndividuals, sigma, omega, data, nSamplesEach=1):
  #     self.data = np.array(data).flatten()
  #     self.nIndividuals = nIndividuals
  #     self.sigma = sigma
  #     self.omega = omega
  #     self.nSamplesEach = nSamplesEach
  #     assert len(data) == nIndividuals
