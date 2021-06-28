#!/usr/bin/env python
import numpy as np
import korali

# 1-d problem
def model(p):
  comm = korali.getWorkerMPIComm()
  rank = comm.Get_rank()
  print("My rank is: " + str(rank))
  x = p["Parameters"][0]
  p["F(x)"] = -0.5 * x * x

