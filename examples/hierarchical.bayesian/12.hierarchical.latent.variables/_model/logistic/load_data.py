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

    self.error = "ind"  # no other choice than individual errors
    self.error_model = "constant"  # might add "proportional" option in addition to "constant"

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

    self.data = np.array(self.data)  # to simplify extraction of x and y

    x_vals = [[] for _ in range(self.nIndividuals)]
    y_vals = [[] for _ in range(self.nIndividuals)]
    for i in range(self.nIndividuals):
      # data: (nInd x nPoints x nDim), with nDim = 3
      # We discard the first dimension: ID
      x_vals[i] = self.data[i, :, 1:2].tolist()  # a list of lists
      y_vals[i] = self.data[i, :, 2].tolist()  # a list, one value per datapoint
    self.x_values = x_vals
    self.y_values = y_vals

    self.data = [d.tolist() for d in self.data
                ]  # Korali expects lists as data (might change in the future)

    N = 4
    self.nLatentSpaceDimensions = 4
    self.sigma = 1 * np.eye(N)
