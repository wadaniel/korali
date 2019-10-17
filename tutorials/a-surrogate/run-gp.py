#!/usr/bin/env python3
import os
import sys

import numpy as np
import korali

k = korali.initialize()

k["Problem"]["Type"] = "Evaluation/GaussianProcess"
k["Problem"]["Covariance Function"] = "George"

# Configuring CMA-ES parameters
k["Solver"]["Type"] = "Optimizer/Rprop"
k["Solver"]["Termination Criteria"]["Max Generations"] = 20
k["Solver"]["Termination Criteria"]["Parameter Relative Tolerance"] = 1e-8


k.run()
# k.dry()
