#!/usr/bin/env python3
import os
import sys
sys.path.append('../_model')
from env import *

target = 0.0
<<<<<<< HEAD
outfile = "observations2-vracer.csv"
=======
outfile = "observations-vracer.csv"
>>>>>>> caching

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
e["Solver"]["Experiences Between Policy Updates"] = 5
e["Solver"]["Cache Persistence"] = 500

e["Solver"]["Refer"]["Target Off Policy Fraction"] = 0.1
e["Solver"]["Refer"]["Cutoff Scale"] = 4.0

### Defining the configuration of replay memory

<<<<<<< HEAD
e["Solver"]["Experience Replay"]["Start Size"] = 65536
e["Solver"]["Experience Replay"]["Maximum Size"] = 131072
=======
e["Solver"]["Experience Replay"]["Start Size"] = 1000
e["Solver"]["Experience Replay"]["Maximum Size"] = 10000
>>>>>>> caching

## Defining Neural Network Configuration for Policy and Critic into Critic Container

e["Solver"]["Critic"]["Discount Factor"] = 0.99
e["Solver"]["Critic"]["Learning Rate"] = 1e-4
e["Solver"]["Critic"]["Mini Batch Size"] = 256

e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Elementwise/Linear"

e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Node Count"] = 128
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Elementwise/Tanh"

e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Node Count"] = 128
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Elementwise/Tanh"

e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Elementwise/Linear"

### Defining Termination Criteria

<<<<<<< HEAD
e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = 499
=======
e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = 490
e["Solver"]["Termination Criteria"]["Max Generations"] = 2500
>>>>>>> caching

### Setting file output configuration

e["File Output"]["Enabled"] = False

### Running Experiment

k.run(e)

<<<<<<< HEAD
### Recording Observations (5 restarts a 100 steps)

print('[Korali] Done training. Now running learned policy to produce observations..')
=======
### Recording Observations

print('[Korali] Done training. Now running learned policy to produce observations.')
>>>>>>> caching

states = []
actions = []

cart = CartPole(0.0)

state = cart.getState().tolist()
done = False
<<<<<<< HEAD

restarts = 0
while restarts < 5:
    cart.reset()
    step = 0
    while not done and step < 100:
     
     action = e.getAction(state)
     
     states.append(state)
     actions.append(action)
     
     done = cart.advance(action)

     reward = cart.getReward()
     step = step + 1

     state = cart.getState().tolist()

    if done:
        print('[Korali] Policy failed during episode roll out no {0} at step {1}!!!'.format(restarts+1, step+1))
        sys.exit(-1)
=======
step = 0
while not done and step < 100:
 
 action = e.getAction(state)
 
 states.append(state)
 actions.append(action)
 
 cart.advance(action)

 reward = cart.getReward()
 step = step + 1

 state = cart.getState().tolist()
>>>>>>> caching

### Creating Output

print('[Korali] Finished recording observations. Writing file {}..'.format(outfile))
with open(outfile, 'w') as f:
    for i in range(len(states)):
        f.write(", ".join(str(s) for s in states[i]))
        f.write(", ")
        f.write(str(actions[i][0]))
        f.write("\n")
