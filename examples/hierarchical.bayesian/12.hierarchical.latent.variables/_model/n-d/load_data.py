import numpy as np


def extr(line, toint=True, tofloat=False):
  ''' Helper for extracting file contents. '''
  result = line.strip("\n ").split(" ")
  if tofloat:
    result = [float(r) for r in result]
  elif toint:
    result = [int(r) for r in result]
  return result




class PopulationData():

  def __init__(self):
    self.nIndividuals = None
    self.nDimensions = None
    self.maxNSamples = None
    self.nSamplesEach = []
    self.sigma = None
    self.dNormal = None
    self.dLognormal = None
    self.dLogitnormal = None
    self.data = None

    filenm = "data_advanced.in"
    print(f"Loading data from {filenm} ... ")

    # Must be run from the example top directory, i.e. the place where _data and _model are subdirectories.
    with open(f'_data/n-d/{filenm}', "r") as fd:
      firstline = fd.readline().strip("\n ").split(" ")
      self.nIndividuals, self.nDimensions, self.maxNSamples = map(
          int, firstline[:3])
      self.sigma = float(firstline[3])
      self.dNormal, self.dLognormal, self.dLogitnormal = map(int, firstline[4:])
      self.data = [[] for _ in range(self.nIndividuals)]
      self.nSamplesEach = np.zeros((self.nIndividuals))

      for i in range(self.nIndividuals):
        c, numSamples = fd.readline().strip("\n ").split(" ")
        numSamples = int(numSamples)
        self.nSamplesEach[i] = numSamples
        assert c == "N"
        for j in range(numSamples):
          self.data[i].append(extr(fd.readline(), tofloat=True))

