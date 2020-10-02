#!/usr/bin/env python3
import os
import sys
sys.path.append('./_model')
from env import *

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining the Cartpole problem's configuration

e["Problem"]["Type"] = "Reinforcement Learning / Discrete"
e["Problem"]["Possible Actions"] = [ [ -10.0 ], [ -9.0 ], [ -8.0 ], [ -7.0 ], [ -6.0 ], [ -5.0 ], [ -4.0 ], [ -3.0 ], [ -2.0 ], [ -1.0 ], [0.0],
                                     [  10.0 ], [  9.0 ], [  8.0 ], [  7.0 ], [  6.0 ], [  5.0 ], [  4.0 ], [  3.0 ], [  2.0 ], [  1.0 ]  ]
e["Problem"]["Environment Function"] = env
e["Problem"]["Action Repeat"] = 1
e["Problem"]["Actions Between Policy Updates"] = 1

e["Variables"][0]["Name"] = "Cart Position"
e["Variables"][0]["Type"] = "State"

e["Variables"][1]["Name"] = "Cart Velocity"
e["Variables"][1]["Type"] = "State"

e["Variables"][2]["Name"] = "Pole Omega"
e["Variables"][2]["Type"] = "State"

e["Variables"][3]["Name"] = "Pole Cos(Angle)"
e["Variables"][3]["Type"] = "State"

e["Variables"][4]["Name"] = "Pole Sin(Angle)"
e["Variables"][4]["Type"] = "State"

e["Variables"][5]["Name"] = "Force"
e["Variables"][5]["Type"] = "Action"

### Configuring ACER hyperparameters

e["Solver"]["Type"] = "Agent / ACER / Discrete"
e["Solver"]["Importance Weight Truncation"] = 5.0
e["Solver"]["Trajectory Size"] = 1000
e["Solver"]["Discount Factor"] = 0.995
e["Solver"]["Off Policy Updates"] = 8
e["Solver"]["Optimization Steps Per Trajectory"] = 1

### Defining Experience Replay configuration

e["Solver"]["Experience Replay"]["Start Size"] = 1500
e["Solver"]["Experience Replay"]["Maximum Size"] = 500000

## Defining Q-Critic and Action-selection (policy) optimizers

e["Solver"]["Critic"]["Optimizer"]["Type"] = "Optimizer/Adam"
e["Solver"]["Critic"]["Optimizer"]["Eta"] = 0.001

e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Node Count"] = 5
e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Elementwise/Linear"
e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Batch Normalization"]["Enabled"] = False

e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Node Count"] = 32
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Elementwise/Tanh"
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Batch Normalization"]["Enabled"] = True

e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Node Count"] = 32
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Elementwise/Tanh"
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Batch Normalization"]["Enabled"] = True

e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Node Count"] = 2
e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Elementwise/Linear"
e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Batch Normalization"]["Enabled"] = True

e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Weight Initialization Scaling"] = 0.000000001

e["Solver"]["Normalization Steps"] = 32
e["Solver"]["Normalization Batch Size"] = 32

## Defining Policy Configuration

e["Solver"]["Policy"]["Optimizer"]["Type"] = "Optimizer/Adam"
e["Solver"]["Policy"]["Optimizer"]["Termination Criteria"]["Min Gradient Norm"] = -1.0
e["Solver"]["Policy"]["Optimizer"]["Eta"] = 0.001

e["Solver"]["Policy"]["Trust Region"]["Enabled"] = True
e["Solver"]["Policy"]["Trust Region"]["Divergence Constraint"] = 1.0
e["Solver"]["Policy"]["Trust Region"]["Adoption Rate"] = 0.99

e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense"
e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Node Count"] = 5
e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Elementwise/Linear"

e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Type"] = "Layer/Dense"
e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Node Count"] = 32
e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Elementwise/Tanh"

e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Dense"
e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Node Count"] = 32
e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Elementwise/Tanh"

e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Type"] = "Layer/Dense"
e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Node Count"] = 1
e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Softmax" 

### Defining Termination Criteria

e["Solver"]["Training Reward Threshold"] = 490
e["Solver"]["Policy Testing Episodes"] = 20
e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = 490

### Setting file output configuration

e["File Output"]["Frequency"] = 100000

### Running Experiment
k.run(e)
