#!/usr/bin/env python3
import os
import sys
sys.path.append('../_model')
from env import *

target = 0.0
outfile = "observations-vracer.csv"

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

envp = lambda s : env(s,target)

### Defining the Cartpole problem's configuration

e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = envp
e["Problem"]["Training Reward Threshold"] = 490
e["Problem"]["Policy Testing Episodes"] = 10
e["Problem"]["Actions Between Policy Updates"] = 5

e["Variables"][0]["Name"] = "Cart Position"
e["Variables"][0]["Type"] = "State"

e["Variables"][1]["Name"] = "Cart Velocity"
e["Variables"][1]["Type"] = "State"

e["Variables"][2]["Name"] = "Pole Angle"
e["Variables"][2]["Type"] = "State"

e["Variables"][3]["Name"] = "Pole Angular Velocity"
e["Variables"][3]["Type"] = "State"

e["Variables"][4]["Name"] = "Force"
e["Variables"][4]["Type"] = "Action"
e["Variables"][4]["Lower Bound"] = -10.0
e["Variables"][4]["Upper Bound"] = +10.0

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Experiences Between Policy Updates"] = 10
e["Solver"]["Experiences Per Generation"] = 500
e["Solver"]["Cache Persistence"] = 100

### Defining the configuration of replay memory

e["Solver"]["Experience Replay"]["Start Size"] = 1000
e["Solver"]["Experience Replay"]["Maximum Size"] = 10000
e["Solver"]["Experience Replay"]["REFER"]["Enabled"] = True
e["Solver"]["Experience Replay"]["REFER"]["Cutoff Scale"] = 4.0
e["Solver"]["Experience Replay"]["REFER"]["Target"] = 0.1
e["Solver"]["Experience Replay"]["REFER"]["Initial Beta"] = 0.3
e["Solver"]["Experience Replay"]["REFER"]["Annealing Rate"] = 5e-7

## Defining Neural Network Configuration for Policy and Critic into Critic Container

e["Solver"]["Discount Factor"] = 0.99
e["Solver"]["Learning Rate"] = 1e-4
e["Solver"]["Mini Batch Size"] = 32

### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 32

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 32

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

### Defining Termination Criteria

e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = 490
e["Solver"]["Termination Criteria"]["Max Generations"] = 2500

### Setting file output configuration

e["File Output"]["Enabled"] = False

### Running Experiment

k.run(e)

### Recording Observations

print('[Korali] Done training. Now running learned policy to produce observations.')


### Now testing policy, dumping trajectory results

# Adding custom setting to run the environment dumping the state files during testing
e["Problem"]["Custom Settings"]["Dump Frequency"] = 0.1
e["Problem"]["Custom Settings"]["Dump Path"] = "_testingResults"

e["File Output"]["Path"] = "_testingResults"
e["Solver"]["Testing"]["Policy"] = e["Solver"]["Best Training Hyperparamters"]
e["Solver"]["Mode"] = "Testing"
for i in range(10):
    e["Solver"]["Testing"]["Sample Ids"][i] = i

k.run(e)

printf("[Korali] Finished testing.")


#states = []
#actions = []

#cart = CartPole(0.0)

#state = cart.getState().tolist()
#done = False
#step = 0
#while not done and step < 100:
 
# action = e.getAction(state)
 
# states.append(state)
# actions.append(action)
 
# cart.advance(action)

# reward = cart.getReward()
# step = step + 1

# state = cart.getState().tolist()

### Creating Output

#print('[Korali] Finished recording observations. Writing file {}..'.format(outfile))
#with open(outfile, 'w') as f:
#    for i in range(len(states)):
#        f.write(", ".join(str(s) for s in states[i]))
#        f.write(", ")
#        f.write(str(actions[i][0]))
#        f.write("\n")
