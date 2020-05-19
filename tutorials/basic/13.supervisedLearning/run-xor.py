#!/usr/bin/env python3
import os
import sys
import korali

# Defining Training Sets
tInputSet  = [ [ 0.0, 0.0 ], [ 0.0, 1.0 ], [ 1.0, 0.0 ], [ 1.0, 1.0 ]  ]

# Describing Korali Experiment
e = korali.Experiment()

e["Problem"]["Type"] = "Supervised Learning"
e["Solver"]["Type"] = "Neural Network"

e["Solver"]["Operation"] = "Forward"
e["Solver"]["Batch Input"]  = tInputSet 

e["Solver"]["Layers"][0]["Type"] = "Input"
e["Solver"]["Layers"][0]["Node Count"] = 2
e["Solver"]["Layers"][0]["Activation Function"] = "Identity"

e["Solver"]["Layers"][1]["Type"] = "Dense"
e["Solver"]["Layers"][1]["Node Count"] = 2
e["Solver"]["Layers"][1]["Activation Function"] = "ReLU"
e["Solver"]["Layers"][1]["Weights"] = [ [ 1.0, 1.0 ], [ 1.0, 1.0 ] ]
e["Solver"]["Layers"][1]["Bias"] = [ 0.0, -1.0 ] 

e["Solver"]["Layers"][2]["Type"] = "Output"
e["Solver"]["Layers"][2]["Node Count"] = 1
e["Solver"]["Layers"][2]["Activation Function"] = "Identity"
e["Solver"]["Layers"][2]["Weights"] = [ [ 1.0, -2.0 ] ]
e["Solver"]["Layers"][2]["Bias"] = [ 0.0 ]

k = korali.Engine()
#k.run(e)

