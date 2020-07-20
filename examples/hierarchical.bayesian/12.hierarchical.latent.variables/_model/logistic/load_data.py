import numpy as np


def extr(line, toint=True, tofloat=False, delim=" "):
  ''' Helper for extracting file contents. '''
  result = line.strip("\n ").split(delim)
  if tofloat:
    result = [float(r) for r in result]
  elif toint:
    result = [int(r) for r in result]
  return result


class LogisticData():

  def __init__(self):
    self.nIndividuals = None
    self.nDataTotal = None
    self.nDataDimensions = 1
    self.nLatentSpaceDimensions = None
    # par_transf = [0 0 0]; --> three variables, all normal
    self.dNormal = None
    self.dLognormal = None
    self.dProbitnormal = None
    self.dLogitnormal = None

    self.error = "ind"  # no other choice than individual errors
    self.error_model = "constant"  # might add "proportional" option

    self.nSamplesEach = []
    self.data = []
    # Must be run from the example top directory, i.e. the place where _data and _model are subdirectories
    filename = '_data/logistic/all_data.txt'
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
    self.beta = [1, 1, 1, 1]
    self.omega = 100 * np.diag([1, 1, 1, 1])
    # self.alpha = 1
    self.Nmp = len(self.beta) - 1
    self.N = len(self.beta)
    self.nLatentSpaceDimensions = len(self.beta)
    self.omega_chol = np.linalg.cholesky(self.omega)
    self.sigma = 1 * np.eye(self.N)

    self.transf = np.array([0, 0, 0])
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
