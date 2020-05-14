#!/usr/bin/env python3
import os
import sys
import korali

# Defining Training Sets
tInputSet  = [ [ 0.2, 0.3 ], [ 0.7, 0.2 ]  ]
tOutputSet = [ [ 15.3 ], [ 35.3 ]  ]

# Describing Korali Experiment
e = korali.Experiment()

e["Problem"]["Type"] = "Supervised Learning"
e["Solver"]["Type"] = "Neural Network"

e["Solver"]["Operation"] = "Train"
e["Solver"]["Engine Kind"] = "CPU"
e["Solver"]["Optimizer"] = "CMAES"
e["Solver"]["Termination Criteria"]["Max Generations"] = 1
e["Solver"]["Weight Initialization"] = "Xavier"
e["Solver"]["Loss Function"] = "Mean Squared Error" 

e["Solver"]["Training Set Input"]  = tInputSet
e["Solver"]["Training Set Output"] = tOutputSet

e["Solver"]["Layers"][0]["Type"] = "Input"
e["Solver"]["Layers"][0]["Node Count"] = 2
e["Solver"]["Layers"][0]["Activation Function"] = "Identity"

e["Solver"]["Layers"][1]["Type"] = "Dense"
e["Solver"]["Layers"][1]["Node Count"] = 80
e["Solver"]["Layers"][1]["Activation Function"] = "SoftSign"

e["Solver"]["Layers"][2]["Type"] = "Dense"
e["Solver"]["Layers"][2]["Node Count"] = 60
e["Solver"]["Layers"][2]["Activation Function"] = "SoftMax"

e["Solver"]["Layers"][3]["Type"] = "Output"
e["Solver"]["Layers"][3]["Node Count"] = 1
e["Solver"]["Layers"][3]["Activation Function"] = "Identity"

k = korali.Engine()
k.run(e)

