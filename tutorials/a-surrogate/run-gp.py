#!/usr/bin/env python3
import os
import sys

import numpy as np
import korali


f = open( 'data/simple.txt', "r" )
data = [ [int(n) for n in line.split()]  for line in f]
f.close()
x = [row[:-1] for row in data]
y = [row[ -1] for row in data]




n = 40;
x = [[i] for i in range(n)]
y = [2*i+1 for i in range(n)]

k = korali.initialize()

k["Problem"]["Type"] = "Evaluation/GaussianProcess"
k["Problem"]["Covariance Function"] = "CovSum ( CovSEiso, CovNoise)"
k["Problem"]["X Data"] = x
k["Problem"]["Y Data"] = y

# Configuring CMA-ES parameters
k["Solver"]["Type"] = "Optimizer/Rprop"
k["Solver"]["Termination Criteria"]["Max Generations"] = 200
k["Solver"]["Termination Criteria"]["Parameter Relative Tolerance"] = 1e-8


k.run()
# k.dry()
