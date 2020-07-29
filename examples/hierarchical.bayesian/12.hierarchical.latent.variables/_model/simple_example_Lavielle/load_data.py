import numpy as np


def extr(line, toint=True, tofloat=False):
  result = line.strip("\n ").split(" ")
  if tofloat:
    result = [float(r) for r in result]
  elif toint:
    result = [int(r) for r in result]
  return result


class SimplePopulationData():

  def __init__(self):
    self.nIndividuals = None
    self.sigma = None
    self.omega = None
    self.nSamplesEach = 1
    self.data = []

    print("Loading data from data_925.in ... \n")
    # Must be run from the example top directory, i.e. the place where _data and _model are subdirectories
    with open("_data/simple_example_Lavielle/data_925.in", "r") as fd:
      firstline = fd.readline().strip("\n ").split(" ")
      self.nIndividuals, self.nSamplesEach = map(int, firstline[:2])
      self.sigma, self.omega = map(float, firstline[2:])
      for i in range(self.nIndividuals):
        values = extr(fd.readline(), tofloat=True)
        assert len(
            values
        ) == 1, "need to rewrite this to accomodate multiple samples per individual (or multiple dimensions) - use another class for that"
        self.data.append(
            [values])  # one data point of one dimension for each individual

    self.data = np.array(self.data)

  def reset_to(self, nIndividuals, sigma, omega, data, nSamplesEach=1):
    self.data = np.array(data).flatten()
    self.nIndividuals = nIndividuals
    self.sigma = sigma
    self.omega = omega
    self.nSamplesEach = nSamplesEach
    assert len(data) == nIndividuals


#
# class PopulationData():
#     def __init__(self):
#         self.nIndividuals = None
#         self.nDimensions = None
#         self.maxNSamples = None
#         self.nSamplesEach = []
#         self.sigma = None
#         self.dNormal = None
#         self.dLognormal = None
#         self.dLogitnormal = None
#         self.data = None
#
#         filenm = "data_advanced.in"
#         print(f"Loading data from {filenm} ... ")
#         with open(f'_data/{filenm}', "r") as fd:
#             firstline = fd.readline().strip("\n ").split(" ")
#             self.nIndividuals, self.nDimensions, self.maxNSamples = map(int, firstline[:3])
#             self.sigma = float(firstline[3])
#             self.dNormal, self.dLognormal, self.dLogitnormal = map(int, firstline[4:])
#             self.data = [[] for _ in range(self.nIndividuals)]
#             self.nSamplesEach = np.zeros((self.nIndividuals))
#
#             for i in range(self.nIndividuals):
#                 c, numSamples = fd.readline().strip("\n ").split(" ")
#                 numSamples = int(numSamples)
#                 self.nSamplesEach[i] = numSamples
#                 assert c == "N"
#                 self.data[i] = np.zeros((numSamples, self.nDimensions))
#                 for j in range(numSamples):
#                     self.data[i][j] = extr(fd.readline(), tofloat=True)
#
#         # test
#         print(self.data[0].shape)
#         print(len(self.data))
