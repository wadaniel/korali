#!/usr/bin/env python3
import os
import sys
sys.path.append('./model/')
from integrand import *
import numpy as np

# Variables
x = np.linspace (0.0,5.0, 1000, dtype=np.float64) #grid for quadrature
y = np.linspace (0.0,5.0, 1000, dtype=np.float64) #grid for quadrature

# Creating new experiment
import korali
e = korali.Experiment()

e["Problem"]["Type"] = "Integration"
e["Problem"]["Integrand"] = lambda modelData: integrand(modelData)
e["Problem"]["Integration Method"] = "Monte Carlo"#"Rectangle" #"Trapezoidal", "Simpson", "Monte Carlo", "Custom"
e["Problem"]["Number Of Samples"] = 1000

e["Distributions"][0]["Name"] = "Uniform"
e["Distributions"][0]["Type"] = "Univariate/Uniform"
e["Distributions"][0]["Minimum"] = 0.0
e["Distributions"][0]["Maximum"] = +1.0

e["Variables"][0]["Name"] = "x"
e["Variables"][0]["Number Of Gridpoints"] = 10
e["Variables"][0]["Lower Bound"] = 0.0
e["Variables"][0]["Upper Bound"] = 1.0
e["Variables"][0]["Prior Distribution"] = "Uniform" 

e["Variables"][1]["Name"] = "y"
e["Variables"][1]["Lower Bound"] = 0.0
e["Variables"][1]["Upper Bound"] = 1.0
e["Variables"][1]["Number Of Gridpoints"] = 10
e["Variables"][1]["Prior Distribution"] = "Uniform" 

e["Variables"][2]["Name"] = "z"
e["Variables"][2]["Lower Bound"] = 0.0
e["Variables"][2]["Upper Bound"] = 1.0
e["Variables"][2]["Number Of Gridpoints"] = 10
e["Variables"][2]["Prior Distribution"] = "Uniform" 


e["Solver"]["Type"] = "Integrator"
#e["Solver"]["Executions Per Generation"] = 0
e["Console Output"]["Verbosity"] = "Detailed"

# Starting Korali's Engine and running experiment
k = korali.Engine()
k.run(e)

