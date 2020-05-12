#!/usr/bin/env python3
import os
import sys
import korali

# Defining Training Set [[x1,y1,q1],[x2,y2,q2]]
tSet = [ [ 0.2, 0.3, 15.3 ], [ 0.7, 0.2, 35.3 ]  ]

# Describing Korali Experiment
e = korali.Experiment()

e["Problem"]["Type"] = "Supervised Learning"
e["Problem"]["Training Set"] = tSet 
e["Problem"]["Loss Function"] = "Mean Squared Error"

# Configuring Input/Output Variables
e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Type"] = "Input"

e["Variables"][1]["Name"] = "Y"
e["Variables"][1]["Type"] = "Input"

e["Variables"][2]["Name"] = "Q"
e["Variables"][2]["Type"] = "Output"

# Configuring Solver

e["Solver"]["Type"] = "Neural Network"
e["Solver"]["Operation"] = "Train"
e["Solver"]["Backend"] = "oneDNN"
e["Solver"]["Optimizer"] = "CMAES"
e["Solver"]["Termination Criteria"]["Max Generations"] = 1
e["Solver"]["Weight Initialization"] = "Xavier"

e["Solver"]["Layers"][0]["Type"] = "Input"
e["Solver"]["Layers"][0]["Nodes"] = 2
e["Solver"]["Layers"][0]["Activation Function"] = "Identity"

e["Solver"]["Layers"][1]["Type"] = "Dense"
e["Solver"]["Layers"][1]["Nodes"] = 80
e["Solver"]["Layers"][1]["Activation Function"] = "SoftSign"

e["Solver"]["Layers"][2]["Type"] = "Dense"
e["Solver"]["Layers"][2]["Nodes"] = 60
e["Solver"]["Layers"][2]["Activation Function"] = "SoftMax"

e["Solver"]["Layers"][3]["Type"] = "Output"
e["Solver"]["Layers"][3]["Nodes"] = 1
e["Solver"]["Layers"][3]["Activation Function"] = "Identity"

k = korali.Engine()
k.run(e)

# Defining Forward Set [[x1,y1],[x2,y2]]
fSet = [ [ 0.4, 0.9 ], [ 0.1, 0.0 ]  ]

e["Problem"]["Forward Set"] = fSet 
e["Solver"]["Operation"] = "Forward"

k.run(e)

#outputs = e["Results"]["Outputs"]