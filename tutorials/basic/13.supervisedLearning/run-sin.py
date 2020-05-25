#!/usr/bin/env python3
import os
import sys
import korali
import numpy as np

# Defining Training Sets

tInputSet  = np.random.uniform(0,2*np.pi,100)
tOutputSet = np.sin(tInputSet)

tInputSet = [ [ i ] for i in tInputSet.tolist() ]
tOutputSet = [ [ i ] for i in tOutputSet.tolist() ]  

# Describing Neural Network-based Model
def neuralModel(sample, input, output):
 e = korali.Experiment()
 
 e["Problem"]["Type"] = "Supervised Learning"
 e["Solver"]["Type"] = "Neural Network"
 
 p = sample["Parameters"]
 
 e["Solver"]["Layers"][0]["Type"] = "Input"
 e["Solver"]["Layers"][0]["Node Count"] = 1
 e["Solver"]["Layers"][0]["Node Values"] = input
 e["Solver"]["Layers"][0]["Activation Function"] = "Identity"
 
 e["Solver"]["Layers"][1]["Type"] = "Dense"
 e["Solver"]["Layers"][1]["Node Count"] = 4
 e["Solver"]["Layers"][1]["Weights"] = [ [ p[0] ], [ p[1] ], [ p[2] ], [ p[3] ] ] 
 e["Solver"]["Layers"][1]["Bias"] = [ p[4], p[5], p[6], p[7] ]
 e["Solver"]["Layers"][1]["Activation Function"] = "Tanh"
 
 e["Solver"]["Layers"][2]["Type"] = "Dense"
 e["Solver"]["Layers"][2]["Node Count"] = 4
 e["Solver"]["Layers"][2]["Weights"] = [ [ p[8], p[9], p[10], p[11] ], [ p[12], p[13], p[14], p[15] ], [ p[16], p[17], p[18], p[19] ], [ p[20], p[21], p[22], p[23] ]  ]
 e["Solver"]["Layers"][2]["Bias"] = [ p[24], p[25], p[26], p[27] ]
 e["Solver"]["Layers"][2]["Activation Function"] = "Tanh"
 
 e["Solver"]["Layers"][3]["Type"] = "Output"
 e["Solver"]["Layers"][3]["Node Count"] = 1
 e["Solver"]["Layers"][3]["Weights"] = [ [ p[28],  p[29], p[30],  p[31] ] ]
 e["Solver"]["Layers"][3]["Bias"] = [ p[32] ]
 e["Solver"]["Layers"][3]["Activation Function"] = "Identity"
 
 e["Solver"]["Termination Criteria"]["Max Generations"] = 1
 e["File Output"]["Enabled"] = False
 e["Console Output"]["Verbosity"] = "Silent"
 
 k = korali.Engine()
 k.run(e)

 ## Calculating MSE
 
 nnOut = e["Solver"]["Layers"][3]["Node Values"] 
 
 meanSquareError = 0
 for i in range(len(nnOut)):
  for j in range(len(nnOut[i])):
   error = output[i][j] - nnOut[i][j]
   meanSquareError += error*error
 meanSquareError = meanSquareError / (len(nnOut) * len(nnOut[0]))
 
 sample["F(x)"] = -meanSquareError
 
# Creating new experiment
e = korali.Experiment()

# Configuring Problem
e["Problem"]["Type"] = "Optimization/Stochastic"
e["Problem"]["Objective Function"] = lambda x : neuralModel(x, tInputSet, tOutputSet) 

# Defining the problem's variables.
for i in range(33):
 e["Variables"][i]["Name"] = "X" + str(i)
 e["Variables"][i]["Lower Bound"] = -10.0
 e["Variables"][i]["Upper Bound"] = +10.0

# Configuring CMA-ES parameters
e["Solver"]["Type"] = "CMAES"
e["Solver"]["Population Size"] = 50 
e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-9

# Running Korali
k = korali.Engine()
#k.run(e)
