#!/usr/bin/env python
import numpy as np
import sys
import csv
import korali

sys.path.append('./_rl_model/_model')
from env import *


# 1-d problem
def model(p):
  x = p["Parameters"][0]
  p["F(x)"] = -0.5 * x * x

# multi dimensional problem (rosenbrock)
def negative_rosenbrock(p):
    x = p["Parameters"]
    dim = len(x)
    res = 0.
    for i in range(dim-1):
        res += 100*(x[i+1]-x[i]**2)**2+(1-x[i])**2

    p["F(x)"] = -res


# rl cartpole with unknown reward
def rl_cartpole_vracer(p):

    k = korali.Engine()
    e = korali.Experiment()

    target = p["Parameters"][0]
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

    e["Solver"]["Experience Replay"]["Start Size"] = 1000
    e["Solver"]["Experience Replay"]["Maximum Size"] = 10000

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

    e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = 490
    e["Solver"]["Termination Criteria"]["Max Generations"] = 2500

    ### Setting file output configuration

    e["File Output"]["Enabled"] = False
    e["Console Output"]["Verbosity"] = "Silent"

    ### Running Experiment

    k.run(e)

    ## Read observations

    states = []
    obsactions = []

    obsfile = '_rl_model/data/observations-vracer.csv'

    with open(obsfile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for row in csv_reader:
            states.append(row[:4])
            obsactions.append(row[4])

    ### Evaluating sum of squarred errors

    suml2 = 0.0

    for i, state in enumerate(states):
        action = e.getAction(state)
        l2 = np.sum(np.power((np.array(obsactions[i])-np.array(action)),2))
        suml2 += l2
    
    p["F(x)"] = -suml2 # maximize
