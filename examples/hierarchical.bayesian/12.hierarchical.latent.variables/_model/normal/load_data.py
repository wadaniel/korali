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


class NormalData:
  '''
    Reads the data (x and y values) for the 'normal' example, as well as other parameters of the example,
     from '_data/normal/all_data.txt'.
  '''

  def __init__(self, datafile=None):
    self.nIndividuals = None
    self.nDataTotal = None
    self.nLatentSpaceDimensions = None

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

    # ** Extract x and y values from the data **
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
                ]  # Korali expects data as lists (might change in the future)

    N = 2
    self.nLatentSpaceDimensions = 2
    self.sigma = 1 * np.eye(N)
